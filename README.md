# E-Health Voice Assistant — Python Web App

A simple Flask web app: tap the mic, speak in Luganda (or another supported
language), and get a spoken + written answer to common health questions.
Speech-to-text and text-to-speech are powered by **Sunbird AI**
(https://docs.sunbird.ai).

## Prerequisites

- Python 3.9+
- **ffmpeg** installed and on your PATH (required for converting browser audio
  to a format Sunbird's STT accepts):
  - Windows: download from https://ffmpeg.org/download.html and add to PATH
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`

## Setup

```bash
cd ehealth-app
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Your Sunbird AI API key is already in `.env` as `SUNBIRD_API_KEY`.
**Do not commit `.env` to version control or share it publicly** — treat it
like a password. If you ever need a new one, generate it via the
`/auth/token` endpoint described in the Sunbird AI docs.

## Run

```bash
python app.py
```

Then open http://localhost:5000 in your browser (Chrome or Edge recommended
— they support the MediaRecorder API used for mic recording).

## How it works

1. **Quick intent buttons** (Malaria, Maternal Health, etc.) call
   `/api/text-query`, which looks up a pre-written answer and asks Sunbird AI's
   `/tasks/tts` endpoint to speak it — no speech recognition involved. Good
   for demoing without a working mic.
2. **Mic button** records audio in the browser, uploads it to
   `/api/voice-query`, which:
   - converts the recording to 16kHz mono WAV,
   - sends it to Sunbird AI's `/tasks/stt` endpoint for transcription,
   - matches the transcript against a small keyword-based intent list
     (`data/intents.json`),
   - sends the matched answer to `/tasks/tts` and returns the audio + text.

## Known limitations (by design, for a first prototype)

- Intent matching is keyword-based, not real NLP — add more keywords to
  `data/intents.json` as you test with real speakers.
- Sunbird's TTS signed URLs expire in ~2 minutes — the app plays them
  immediately, which is fine for this flow, but don't cache them long-term.
- Only 4 sample health topics are included — extend `data/intents.json` with
  more Ministry of Health-approved content as needed.
- Free-tier Sunbird accounts have rate limits — see the Sunbird AI docs for
  current limits if you hit 429 errors.

## Next steps

- Add more health topics and keywords for better matching accuracy.
- Add a `/api/facilities/nearby` endpoint if you want the "Nearest Clinic"
  intent to return real facility data instead of a generic message.
- Consider swapping the keyword matcher for Sunbird AI's `/tasks/language_id`
  + a small classifier once you have more real transcripts to test against.
