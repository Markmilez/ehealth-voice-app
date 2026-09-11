"""
Voice-Enabled Local-Language E-Health App — Python backend (Flask)
Wraps the Sunbird AI API (STT + TTS) and serves a simple web front end.
"""
import os
import json
import tempfile

import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

SUNBIRD_API_KEY = os.environ.get("SUNBIRD_API_KEY", "")
SUNBIRD_BASE_URL = "https://api.sunbird.ai"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the local health-info / intent database
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "intents.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    INTENTS = json.load(f)

# Map speaker_id per language for Sunbird TTS (see docs.sunbird.ai)
SPEAKER_IDS = {
    "lug": 248,  # Luganda
    "ach": 241,  # Acholi
    "teo": 242,  # Ateso
    "nyn": 243,  # Runyankole
    "lgg": 245,  # Lugbara
    "swa": 246,  # Swahili
    "eng": 248,  # fallback: use Luganda voice for English text (Sunbird has no English speaker listed)
}


def sunbird_headers(content_type="application/json"):
    headers = {"Authorization": f"Bearer {SUNBIRD_API_KEY}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def convert_to_wav(file_storage):
    """
    Browser MediaRecorder produces webm/ogg, which Sunbird's STT does not accept
    (mp3/wav/ogg/m4a/aac only, and ogg support can be inconsistent). Convert to
    16kHz mono WAV server-side using ffmpeg (via pydub) for reliable results.
    Requires ffmpeg to be installed on the server (see README).
    """
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        file_storage.save(tmp_in.name)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".webm", ".wav")
    audio = AudioSegment.from_file(tmp_in_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(tmp_out_path, format="wav")

    os.remove(tmp_in_path)
    return tmp_out_path


def transcribe_audio(file_storage, language="lug"):
    """Convert the uploaded audio, then send it to Sunbird AI's /tasks/stt endpoint."""
    wav_path = convert_to_wav(file_storage)
    try:
        with open(wav_path, "rb") as f:
            url = f"{SUNBIRD_BASE_URL}/tasks/stt"
            files = {"audio": ("query.wav", f, "audio/wav")}
            data = {"language": language, "adapter": language}
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {SUNBIRD_API_KEY}"},
                files=files,
                data=data,
                timeout=180,
            )
        resp.raise_for_status()
        return resp.json()
    finally:
        os.remove(wav_path)


def synthesize_speech(text, language="lug"):
    """Send text to Sunbird AI's /tasks/tts endpoint and return the signed audio URL."""
    url = f"{SUNBIRD_BASE_URL}/tasks/tts"
    speaker_id = SPEAKER_IDS.get(language, 248)
    payload = {"text": text, "speaker_id": speaker_id, "temperature": 0.7}
    resp = requests.post(url, headers=sunbird_headers(), json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["output"]["audio_url"]


def match_intent(transcript, language="lug"):
    """Very simple keyword matcher against the local intents database."""
    text = (transcript or "").lower()
    key = "keywords_lg" if language == "lug" else "keywords_en"
    best = None
    for intent in INTENTS:
        for kw in intent.get(key, []) + intent.get("keywords_en", []):
            if kw.lower() in text:
                best = intent
                break
        if best:
            break
    return best


def build_answer(intent, language="lug"):
    if language == "lug":
        return intent.get("answer_lg") or intent.get("answer_en")
    return intent.get("answer_en") or intent.get("answer_lg")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", intents=INTENTS)


@app.route("/api/intents")
def api_intents():
    return jsonify(
        [
            {"id": i["id"], "label_en": i["label_en"], "label_lg": i["label_lg"]}
            for i in INTENTS
        ]
    )


@app.route("/api/text-query", methods=["POST"])
def api_text_query():
    body = request.get_json(force=True)
    intent_id = body.get("intent_id")
    language = body.get("language", "lug")

    intent = next((i for i in INTENTS if i["id"] == intent_id), None)
    if not intent:
        return jsonify({"error": "unknown_intent", "message": "No such intent."}), 404

    answer_text = build_answer(intent, language)

    try:
        audio_url = synthesize_speech(answer_text, language)
    except requests.HTTPError as e:
        return jsonify({"error": "tts_failed", "message": str(e)}), 502

    return jsonify(
        {
            "intent": intent["id"],
            "answer_text": answer_text,
            "audio_url": audio_url,
        }
    )


@app.route("/api/voice-query", methods=["POST"])
def api_voice_query():
    if "audio" not in request.files:
        return jsonify({"error": "missing_audio", "message": "No audio file uploaded."}), 400

    audio_file = request.files["audio"]
    language = request.form.get("language", "lug")

    # 1. Speech-to-text via Sunbird AI
    try:
        stt_result = transcribe_audio(audio_file, language)
    except requests.HTTPError as e:
        return jsonify({"error": "stt_failed", "message": str(e)}), 502

    transcript = stt_result.get("audio_transcription", "")

    # 2. Match transcript to a known intent
    intent = match_intent(transcript, language)
    if not intent:
        return jsonify(
            {
                "error": "no_match",
                "message": "Sorry, I didn't understand that. Please try again.",
                "transcript": transcript,
            }
        ), 200

    answer_text = build_answer(intent, language)

    # 3. Text-to-speech for the answer via Sunbird AI
    try:
        audio_url = synthesize_speech(answer_text, language)
    except requests.HTTPError as e:
        return jsonify({"error": "tts_failed", "message": str(e), "transcript": transcript}), 502

    return jsonify(
        {
            "transcript": transcript,
            "intent": intent["id"],
            "answer_text": answer_text,
            "audio_url": audio_url,
            "was_audio_trimmed": stt_result.get("was_audio_trimmed", False),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
