FROM python:3.12-slim

WORKDIR /srv

# ffmpeg and the DejaVu fonts, for the clip renderer. The slim image has
# neither: without ffmpeg the clip endpoints answer 503, and without a
# TrueType font Pillow silently falls back to a bitmap default and the
# burned-in captions come out looking broken rather than plain.
# --no-install-recommends keeps this to what is actually needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# The highlight pool the bot offers when someone says something nice
# rather than asking a question. Without it that path silently does
# nothing — which is exactly what happened: compliments got silence in
# production while working locally, because data/ was never copied.
COPY data/highlights.json ./data/highlights.json
# episodes.json IS read at runtime now, by the clipper: it is the only
# place the source URL and the per-second segments live, and captions are
# built from those rather than from a fresh transcription. 7MB, parsed
# lazily on the first clip anyone asks for. Retrieval still goes to
# Pinecone and does not touch this.
# Gzipped: the raw file is 7.3MB and every ingest rewrites it, so it stays
# out of git and this 2.3MB copy ships instead. Regenerate it after an
# ingest with scripts/pack_episodes.py, or newly added episodes cannot be
# clipped (they 404 — nothing else is affected).
COPY data/episodes.json.gz ./data/episodes.json.gz
# The exact-token index. Without it every lookup returns nothing and
# search silently loses the names and numbers it was built for.
COPY data/term_index.json ./data/term_index.json
COPY widget ./widget
COPY demo ./demo

# Non-root runtime user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/healthz')"

# Shell form so $PORT (set by Render/Heroku-style hosts) is honored.
CMD uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
