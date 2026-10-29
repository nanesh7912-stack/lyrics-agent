FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AI_LYRICS_HERO_DATA_DIR=/var/data/lyrics-hero \
    AI_LYRICS_HERO_MODEL_CACHE=/var/data/whisper-models

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core fonts-freefont-ttf fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY ai_lyrics_hero ./ai_lyrics_hero
COPY assets ./assets
EXPOSE 8000
CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 600 --bind 0.0.0.0:${PORT:-8000} ai_lyrics_hero.wsgi:app"]
