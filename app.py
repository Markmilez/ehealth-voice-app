"""
Voice-Enabled Local-Language E-Health App
-----------------------------------------
Flask backend integrating the current Sunbird AI API.

Features:
- Local intent database
- Speech-to-Text using Sunbird AI
- Text-to-Speech using Sunbird AI
- Luganda, Acholi, Ateso, Runyankole, Lugbara,
  Swahili and English support
- Browser audio conversion to 16 kHz mono WAV
- Detailed Sunbird API error handling
- Health/diagnostic endpoint
"""

import json
import logging
import os
import tempfile
from datetime import datetime

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for, flash
from flask_login import (
    LoginManager, UserMixin, current_user,
    login_required, login_user, logout_user
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from pydub import AudioSegment


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

SUNBIRD_API_KEY = os.environ.get("SUNBIRD_API_KEY", "").strip()
SUNBIRD_BASE_URL = "https://api.sunbird.ai"

app = Flask(__name__)

# Session signing key — set SECRET_KEY in your .env for production.
# Falls back to a dev-only default so local testing still works.
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-change-me")

# Database: defaults to a local SQLite file for quick local testing.
# For anything deployed (Render, etc.), set DATABASE_URL to a real
# Postgres connection string (e.g. from Neon) so accounts/history
# survive restarts and redeploys.
db_url = os.environ.get("DATABASE_URL", "sqlite:///local.db")
# Render/Heroku-style URLs sometimes start with postgres:// — SQLAlchemy needs postgresql://
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Maximum upload size accepted by this application.
# Adjust if necessary.
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger("sunbird-health-app")


# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class QueryHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    mode = db.Column(db.String(20))          # "voice" or "text"
    language = db.Column(db.String(10))
    transcript = db.Column(db.Text)          # null for text-query (chip) entries
    intent_id = db.Column(db.String(80))     # set for chip queries / local fallback
    answer_text = db.Column(db.Text)
    answer_source = db.Column(db.String(20))  # "local" (voice query) or "chip" (topic button)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def log_history(mode, language, answer_text, transcript=None,
                 intent_id=None, answer_source=None):
    """Save one query/answer pair to the current user's history."""
    if not current_user.is_authenticated:
        return
    entry = QueryHistory(
        user_id=current_user.id,
        mode=mode,
        language=language,
        transcript=transcript,
        intent_id=intent_id,
        answer_text=answer_text,
        answer_source=answer_source,
    )
    db.session.add(entry)
    db.session.commit()


# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------

SUPPORTED_STT_LANGUAGES = {
    "lug": "Luganda",
    "ach": "Acholi",
    "teo": "Ateso",
    "nyn": "Runyankole",
    "lgg": "Lugbara",
    "swa": "Swahili",
    "eng": "English",
    "xog": "Lusoga",
    "kin": "Kinyarwanda",
    "myx": "Lumasaba",
}


# Current Sunbird Orpheus TTS voices.
#
# NOTE: these voice IDs are unverified against Sunbird's current
# voice-discovery endpoint. If a TTS call returns 400 with an
# "invalid voice" message, check the API's voice list and update
# this mapping accordingly.
#
# If a language has no configured voice, we omit "voice" and
# allow Sunbird to choose the default voice for that language.
TTS_VOICES = {
    "lug": "salt_lug_0001",
    "ach": "salt_ach_0001",
    "teo": "salt_teo_0001",
    "nyn": "salt_nyn_0001",
    "lgg": "salt_lgg_0001",
    "swa": "salt_swa_0001",
    "eng": "salt_eng_0001",
}


# ---------------------------------------------------------------------------
# UI localization — translates the site's own interface text (not just
# health answers) into the selected language, via Sunbird AI's translate
# endpoint. English is the source of truth; other languages are translated
# on first request per language and cached in memory for the process
# lifetime (fine for a prototype — restarts just re-translate once).
# ---------------------------------------------------------------------------

UI_STRINGS_EN = {
    "tagline_line1": "Ask about your health,",
    "tagline_line2": "out loud.",
    "hero_sub": "Speak in your language. Get an answer you can hear and read.",
    "history_link": "History",
    "logout_link": "Log out",
    "start_btn": "Start",
    "stop_btn": "Stop",
    "status_default": "Press Start and speak",
    "status_listening": "Listening... press Stop when done",
    "status_thinking": "Thinking...",
    "status_mic_denied": "Microphone access denied or unavailable.",
    "status_error": "Something went wrong. Try again.",
}

# language codes with no dedicated Sunbird translate support are left in
# English rather than silently mistranslated.
TRANSLATABLE_LANGUAGES = {"lug", "ach", "teo", "nyn", "lgg"}

UI_STRINGS_CACHE = {}     # language -> {key: translated string}
INTENT_LABEL_CACHE = {}   # language -> {intent_id: translated label}


def translate_text(text, target_language, source_language="eng"):
    """
    Translate a short piece of text via Sunbird AI's translation endpoint.
    """
    validate_api_key()

    url = f"{SUNBIRD_BASE_URL}/tasks/nllb_translate"
    payload = {
        "source_language": source_language,
        "target_language": target_language,
        "text": text,
    }
    headers = sunbird_headers(content_type="application/json")

    response = requests.post(url, headers=headers, json=payload, timeout=(15, 30))

    if not response.ok:
        error = sunbird_error(response)
        raise SunbirdAPIError(
            "Sunbird translation request failed.",
            status_code=response.status_code,
            response_body=error["body"],
            request_id=error["request_id"],
        )

    result = response.json()
    # Handle both a flat shape and a nested {"output": {...}} shape.
    output = result.get("output", result)
    translated = output.get("translated_text")

    if not translated:
        raise SunbirdAPIError(
            "Sunbird translation response missing translated_text.",
            status_code=response.status_code,
            response_body=result,
        )

    return translated


def get_ui_strings(language):
    """Return the UI string dictionary for the given language, translating
    and caching on first use. Falls back to English per-string on any
    translation failure so the UI never breaks."""

    language = (language or "eng").lower().strip()

    if language == "eng" or language not in TRANSLATABLE_LANGUAGES:
        return UI_STRINGS_EN

    if language in UI_STRINGS_CACHE:
        return UI_STRINGS_CACHE[language]

    translated = {}
    for key, text in UI_STRINGS_EN.items():
        try:
            translated[key] = translate_text(text, language)
        except Exception:
            logger.exception(
                "UI string translation failed key=%s language=%s", key, language
            )
            translated[key] = text

    UI_STRINGS_CACHE[language] = translated
    return translated


def get_intent_labels(language):
    """Return {intent_id: label} for the given language. Uses the curated
    label_lg field for Luganda, translates for other supported languages,
    and falls back to English on failure."""

    language = (language or "eng").lower().strip()

    if language == "eng" or language not in TRANSLATABLE_LANGUAGES:
        return {i["id"]: i["label_en"] for i in INTENTS}

    if language == "lug":
        return {i["id"]: (i.get("label_lg") or i["label_en"]) for i in INTENTS}

    if language in INTENT_LABEL_CACHE:
        return INTENT_LABEL_CACHE[language]

    labels = {}
    for i in INTENTS:
        try:
            labels[i["id"]] = translate_text(i["label_en"], language)
        except Exception:
            logger.exception(
                "Intent label translation failed id=%s language=%s", i["id"], language
            )
            labels[i["id"]] = i["label_en"]

    INTENT_LABEL_CACHE[language] = labels
    return labels


# ---------------------------------------------------------------------------
# Load local health-info / intent database
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "intents.json"
)

try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        INTENTS = json.load(f)

    if not isinstance(INTENTS, list):
        raise ValueError("intents.json must contain a JSON list.")

except FileNotFoundError:
    logger.exception("Could not find intents database: %s", DATA_PATH)
    INTENTS = []

except json.JSONDecodeError:
    logger.exception("Invalid JSON in intents database: %s", DATA_PATH)
    INTENTS = []

except Exception:
    logger.exception("Failed to load intents database.")
    INTENTS = []


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def sunbird_headers(content_type=None):
    """
    Build authentication headers for Sunbird AI.

    For multipart/form-data requests, do NOT manually set
    Content-Type because requests must generate the multipart
    boundary automatically.
    """

    headers = {
        "Authorization": f"Bearer {SUNBIRD_API_KEY}",
        "Accept": "application/json",
    }

    if content_type:
        headers["Content-Type"] = content_type

    return headers


def sunbird_error(response):
    """
    Convert a Sunbird error response into a useful diagnostic string.
    """

    try:
        body = response.json()
    except ValueError:
        body = response.text

    return {
        "status": response.status_code,
        "body": body,
        "request_id": response.headers.get("x-request-id")
        or response.headers.get("X-Request-ID"),
    }


def validate_api_key():
    """
    Check that an API key exists before making a Sunbird request.
    """

    if not SUNBIRD_API_KEY:
        raise RuntimeError(
            "SUNBIRD_API_KEY is missing. "
            "Add it to your .env file."
        )


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def convert_to_wav(file_storage):
    """
    Convert browser-recorded audio to 16 kHz mono WAV.

    Browser MediaRecorder commonly produces WebM/Opus.
    Sunbird accepts WAV and other audio formats, but converting
    to a consistent WAV format makes the STT pipeline more reliable.

    Requires ffmpeg to be installed on the server.
    """

    original_name = file_storage.filename or "recording.webm"

    with tempfile.NamedTemporaryFile(
        suffix=".webm",
        delete=False
    ) as tmp_in:

        file_storage.save(tmp_in.name)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".webm", ".wav")

    try:
        audio = AudioSegment.from_file(tmp_in_path)

        # Sunbird/Whisper-friendly format
        audio = (
            audio
            .set_frame_rate(16000)
            .set_channels(1)
            .set_sample_width(2)
        )

        audio.export(
            tmp_out_path,
            format="wav"
        )

        logger.info(
            "Converted uploaded audio '%s' to WAV: %s",
            original_name,
            tmp_out_path
        )

        return tmp_out_path

    except Exception:
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)

        raise

    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)


# ---------------------------------------------------------------------------
# Custom Sunbird exception
# ---------------------------------------------------------------------------

class SunbirdAPIError(Exception):

    def __init__(
        self,
        message,
        status_code=None,
        response_body=None,
        request_id=None
    ):
        super().__init__(message)

        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        self.request_id = request_id

    def to_dict(self):
        return {
            "error": self.message,
            "sunbird_status": self.status_code,
            "sunbird_response": self.response_body,
            "request_id": self.request_id,
        }


# ---------------------------------------------------------------------------
# Sunbird Speech-to-Text
# ---------------------------------------------------------------------------

def transcribe_audio(file_storage, language="lug"):
    """
    Transcribe uploaded audio using the CURRENT Sunbird AI STT API.

    Current endpoint:
        POST /tasks/audio/transcriptions

    Required multipart fields:
        audio
        language

    Optional:
        timestamps

    IMPORTANT:
    The current API does NOT accept the old "adapter" parameter
    used by the deprecated v1 /tasks/stt endpoint.
    """

    validate_api_key()

    language = (language or "lug").lower().strip()

    if language not in SUPPORTED_STT_LANGUAGES:
        raise ValueError(
            f"Unsupported STT language: {language}. "
            f"Supported languages: {', '.join(SUPPORTED_STT_LANGUAGES)}"
        )

    wav_path = convert_to_wav(file_storage)

    try:
        url = f"{SUNBIRD_BASE_URL}/tasks/audio/transcriptions"

        with open(wav_path, "rb") as audio_file:

            files = {
                "audio": (
                    "query.wav",
                    audio_file,
                    "audio/wav"
                )
            }

            data = {
                "language": language
            }

            headers = sunbird_headers()

            logger.info(
                "Sending STT request to %s language=%s",
                url,
                language
            )

            response = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=(30, 180)
            )

        logger.info(
            "Sunbird STT response: HTTP %s",
            response.status_code
        )

        if not response.ok:
            error = sunbird_error(response)

            logger.error(
                "Sunbird STT failed: %s",
                error
            )

            raise SunbirdAPIError(
                "Sunbird STT request failed.",
                status_code=response.status_code,
                response_body=error["body"],
                request_id=error["request_id"]
            )

        try:
            result = response.json()
        except ValueError as exc:
            raise SunbirdAPIError(
                "Sunbird returned an invalid JSON response.",
                status_code=response.status_code,
                response_body=response.text
            ) from exc

        logger.info("STT transcription received successfully.")

        return result

    except requests.Timeout as exc:
        logger.exception("Sunbird STT request timed out.")

        raise SunbirdAPIError(
            "Sunbird STT request timed out.",
            status_code=504
        ) from exc

    except requests.ConnectionError as exc:
        logger.exception("Could not connect to Sunbird AI.")

        raise SunbirdAPIError(
            "Could not connect to Sunbird AI.",
            status_code=503
        ) from exc

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ---------------------------------------------------------------------------
# Sunbird Text-to-Speech
# ---------------------------------------------------------------------------

def synthesize_speech(text, language="lug"):
    """
    Generate speech using the CURRENT Sunbird AI TTS API.

    Current endpoint:
        POST /tasks/audio/speech

    The current API returns a signed audio URL when
    response_mode='url'.
    """

    validate_api_key()

    language = (language or "lug").lower().strip()

    if not text:
        raise ValueError("Text for TTS cannot be empty.")

    if language not in SUPPORTED_STT_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    url = f"{SUNBIRD_BASE_URL}/tasks/audio/speech"

    payload = {
        "text": text,
        "language": language,
        "response_mode": "url",
        "temperature": 0.6,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "max_tokens": 1200,
    }

    # Only specify a voice when we have explicitly configured one.
    voice = TTS_VOICES.get(language)

    if voice:
        payload["voice"] = voice

    headers = sunbird_headers(content_type="application/json")

    logger.info(
        "Sending TTS request to %s language=%s voice=%s",
        url,
        language,
        voice
    )

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(30, 120)
        )

        logger.info(
            "Sunbird TTS response: HTTP %s",
            response.status_code
        )

        if not response.ok:
            error = sunbird_error(response)

            logger.error("Sunbird TTS failed: %s", error)

            raise SunbirdAPIError(
                "Sunbird TTS request failed.",
                status_code=response.status_code,
                response_body=error["body"],
                request_id=error["request_id"]
            )

        try:
            result = response.json()
        except ValueError as exc:
            raise SunbirdAPIError(
                "Sunbird returned invalid TTS JSON.",
                status_code=response.status_code,
                response_body=response.text
            ) from exc

        audio_url = result.get("audio_url")

        if not audio_url:
            raise SunbirdAPIError(
                "Sunbird TTS response did not contain audio_url.",
                status_code=response.status_code,
                response_body=result
            )

        return result

    except requests.Timeout as exc:
        logger.exception("Sunbird TTS request timed out.")

        raise SunbirdAPIError(
            "Sunbird TTS request timed out.",
            status_code=504
        ) from exc

    except requests.ConnectionError as exc:
        logger.exception("Could not connect to Sunbird TTS.")

        raise SunbirdAPIError(
            "Could not connect to Sunbird TTS.",
            status_code=503
        ) from exc


# ---------------------------------------------------------------------------
# Intent matching
# ---------------------------------------------------------------------------

def match_intent(transcript, language="lug"):
    """
    Very simple keyword matcher against the local intents database.
    """

    text = (transcript or "").lower().strip()

    if not text:
        return None

    key = "keywords_lg" if language == "lug" else "keywords_en"

    best = None

    for intent in INTENTS:

        keywords = []
        keywords.extend(intent.get(key, []) or [])

        # Always allow English keywords as fallback.
        if key != "keywords_en":
            keywords.extend(intent.get("keywords_en", []) or [])

        for kw in keywords:
            if not kw:
                continue
            if kw.lower() in text:
                best = intent
                break

        if best:
            break

    return best


# ---------------------------------------------------------------------------
# Answer builder
# ---------------------------------------------------------------------------

def build_answer(intent, language="lug"):

    if not intent:
        return None

    if language == "lug":
        return intent.get("answer_lg") or intent.get("answer_en")

    return intent.get("answer_en") or intent.get("answer_lg")


ANSWER_TRANSLATION_CACHE = {}  # (language, intent_id) -> translated answer text


def build_localized_answer(intent, language="lug"):
    """
    Return the canned answer for this intent in the requested language.
    Luganda and English use the curated answer_lg/answer_en fields directly
    (fast, no API call). Other supported languages translate the English
    answer via Sunbird AI, cached per (language, intent) so repeat queries
    for the same topic don't re-translate.
    """

    if not intent:
        return None

    language = (language or "lug").lower().strip()

    if language in ("lug", "eng"):
        return build_answer(intent, language)

    if language not in TRANSLATABLE_LANGUAGES:
        return intent.get("answer_en")

    cache_key = (language, intent.get("id"))
    if cache_key in ANSWER_TRANSLATION_CACHE:
        return ANSWER_TRANSLATION_CACHE[cache_key]

    english_answer = intent.get("answer_en")

    try:
        translated = translate_text(english_answer, language)
    except Exception:
        logger.exception(
            "Answer translation failed intent=%s language=%s",
            intent.get("id"), language
        )
        return english_answer

    ANSWER_TRANSLATION_CACHE[cache_key] = translated
    return translated


# ---------------------------------------------------------------------------
# Error response helper
# ---------------------------------------------------------------------------

def api_error_response(error, default_message="API request failed."):

    if isinstance(error, SunbirdAPIError):
        payload = error.to_dict()
        # Do not expose the API key.
        return jsonify(payload), (error.status_code if error.status_code else 502)

    if isinstance(error, ValueError):
        return jsonify({"error": "invalid_request", "message": str(error)}), 400

    logger.exception("Unexpected application error.")

    return jsonify({
        "error": "internal_error",
        "message": default_message
    }), 500


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
@login_required
def index():
    return render_template("index.html", intents=INTENTS, username=current_user.username)


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not username or not password:
            flash("Username and password are required.")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.")
            return render_template("register.html")

        if User.query.filter_by(username=username).first():
            flash("That username is already taken.")
            return render_template("register.html")

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        return redirect(url_for("index"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("index"))

        flash("Invalid username or password.")
        return render_template("login.html")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/history")
@login_required
def history_page():
    entries = (
        QueryHistory.query
        .filter_by(user_id=current_user.id)
        .order_by(QueryHistory.created_at.desc())
        .limit(50)
        .all()
    )
    return render_template("history.html", entries=entries, username=current_user.username)


@app.route("/api/history", methods=["GET"])
@login_required
def api_history():
    entries = (
        QueryHistory.query
        .filter_by(user_id=current_user.id)
        .order_by(QueryHistory.created_at.desc())
        .limit(50)
        .all()
    )
    return jsonify([
        {
            "id": e.id,
            "mode": e.mode,
            "language": e.language,
            "transcript": e.transcript,
            "intent_id": e.intent_id,
            "answer_text": e.answer_text,
            "answer_source": e.answer_source,
            "created_at": e.created_at.isoformat() + "Z",
        }
        for e in entries
    ])


@app.route("/api/intents", methods=["GET"])
@login_required
def api_intents():
    language = request.args.get("language", "eng").lower().strip()
    labels = get_intent_labels(language)
    return jsonify([
        {
            "id": intent.get("id"),
            "label": labels.get(intent.get("id"), intent.get("label_en")),
            "label_en": intent.get("label_en"),
            "label_lg": intent.get("label_lg"),
        }
        for intent in INTENTS
    ])


@app.route("/api/ui-strings", methods=["GET"])
@login_required
def api_ui_strings():
    language = request.args.get("language", "eng").lower().strip()
    return jsonify(get_ui_strings(language))


@app.route("/api/languages", methods=["GET"])
@login_required
def api_languages():
    return jsonify({
        "stt": SUPPORTED_STT_LANGUAGES,
        "tts": TTS_VOICES
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "sunbird_configured": bool(SUNBIRD_API_KEY),
        "sunbird_base_url": SUNBIRD_BASE_URL,
        "intent_count": len(INTENTS),
    })


@app.route("/api/text-query", methods=["POST"])
@login_required
def api_text_query():

    try:
        body = request.get_json(force=True)

        intent_id = body.get("intent_id")
        language = body.get("language", "lug").lower().strip()

        if not intent_id:
            return jsonify({
                "error": "missing_intent_id",
                "message": "intent_id is required."
            }), 400

        intent = next(
            (item for item in INTENTS if item.get("id") == intent_id),
            None
        )

        if not intent:
            return jsonify({
                "error": "unknown_intent",
                "message": "No such intent."
            }), 404

        answer_text = build_localized_answer(intent, language)

        if not answer_text:
            return jsonify({
                "error": "missing_answer",
                "message": "No answer is configured for this intent."
            }), 500

        try:
            tts_result = synthesize_speech(answer_text, language)
        except Exception as exc:
            return api_error_response(exc, "Text-to-speech failed.")

        log_history(
            mode="text",
            language=language,
            answer_text=answer_text,
            intent_id=intent.get("id"),
            answer_source="chip",
        )

        return jsonify({
            "intent": intent.get("id"),
            "answer_text": answer_text,
            "audio_url": tts_result.get("audio_url"),
            "tts": tts_result,
        })

    except Exception as exc:
        return api_error_response(exc, "Text query failed.")


@app.route("/api/voice-query", methods=["POST"])
@login_required
def api_voice_query():

    if "audio" not in request.files:
        return jsonify({
            "error": "missing_audio",
            "message": "No audio file uploaded."
        }), 400

    audio_file = request.files["audio"]

    if not audio_file or not audio_file.filename:
        return jsonify({
            "error": "empty_audio",
            "message": "The uploaded audio file is empty."
        }), 400

    language = request.form.get("language", "lug").lower().strip()

    timestamps = request.form.get("timestamps", "false").lower() in ("true", "1", "yes")

    if language not in SUPPORTED_STT_LANGUAGES:
        return jsonify({
            "error": "unsupported_language",
            "message": (
                f"Unsupported language '{language}'. "
                f"Supported languages: {', '.join(SUPPORTED_STT_LANGUAGES)}"
            )
        }), 400

    # 1. Speech-to-text
    try:
        stt_result = transcribe_audio(audio_file, language)
    except Exception as exc:
        return api_error_response(exc, "Speech-to-text failed.")

    transcript = (stt_result.get("audio_transcription") or "").strip()

    if not transcript:
        return jsonify({
            "error": "empty_transcript",
            "message": "No speech could be recognized from the audio.",
            "stt": stt_result,
        }), 200

    # 2. Match the transcript against the local intent database directly.
    # Luganda and English match on their own keyword lists. Other supported
    # languages are translated to English first (Sunbird AI), then matched
    # on the English keyword list, since that's the language the intent
    # database's keyword lists are richest in.
    answer_source = "local"
    match_text = transcript

    if language not in ("lug", "eng") and language in TRANSLATABLE_LANGUAGES:
        try:
            match_text = translate_text(transcript, "eng", source_language=language)
        except Exception:
            logger.exception(
                "Transcript translation to English failed for language=%s "
                "— matching on the raw transcript instead.",
                language
            )
            match_text = transcript

    match_language = "eng" if language not in ("lug", "eng") else language
    intent = match_intent(match_text, match_language)

    if not intent:
        return jsonify({
            "error": "no_match",
            "message": (
                "Sorry, I didn't understand that. "
                "Please try again or choose a topic below."
            ),
            "transcript": transcript,
            "stt": stt_result,
        }), 200

    answer_text = build_localized_answer(intent, language)

    if not answer_text:
        return jsonify({
            "error": "no_answer",
            "message": (
                "Sorry, I couldn't find an answer to that. "
                "Please try again or choose a topic below."
            ),
            "transcript": transcript,
            "stt": stt_result,
        }), 200

    # 3. Text-to-speech
    try:
        tts_result = synthesize_speech(answer_text, language)
    except Exception as exc:
        return api_error_response(exc, "Text-to-speech failed.")

    # 4. Save to the user's history
    log_history(
        mode="voice",
        language=language,
        transcript=transcript,
        answer_text=answer_text,
        intent_id=intent.get("id") if intent else None,
        answer_source=answer_source,
    )

    # 5. Final response
    return jsonify({
        "transcript": transcript,
        "intent": intent.get("id") if intent else None,
        "answer_text": answer_text,
        "answer_source": answer_source,
        "audio_url": tts_result.get("audio_url"),
        "was_audio_trimmed": stt_result.get("was_audio_trimmed", False),
        "original_duration_minutes": stt_result.get("original_duration_minutes"),
        "duration_seconds": stt_result.get("duration_seconds"),
        "segments": stt_result.get("segments") if timestamps else None,
        "stt": stt_result,
        "tts": tts_result,
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "file_too_large",
        "message": "The uploaded audio file is too large. Maximum size is 100 MB."
    }), 413


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.exception("Unhandled application error: %s", error)
    return jsonify({
        "error": "internal_server_error",
        "message": "An unexpected server error occurred."
    }), 500


with app.app_context():
    db.create_all()


if __name__ == "__main__":

    if not SUNBIRD_API_KEY:
        logger.warning(
            "SUNBIRD_API_KEY is not configured. "
            "Sunbird requests will fail until it is added to the environment."
        )
    else:
        logger.info("SUNBIRD_API_KEY is configured.")

    logger.info("Loaded %d health intents.", len(INTENTS))

    app.run(debug=True, host="0.0.0.0", port=5000)
