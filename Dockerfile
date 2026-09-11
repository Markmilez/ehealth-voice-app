# Base image: slim Python, small footprint
FROM python:3.11-slim

# ffmpeg is required to convert browser mic recordings (webm) to wav
# before sending them to Sunbird AI's STT endpoint
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT at runtime; gunicorn must bind to it (not a hardcoded port)
CMD gunicorn --workers 3 --timeout 120 --bind 0.0.0.0:$PORT app:app
