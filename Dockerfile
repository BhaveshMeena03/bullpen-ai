FROM python:3.12-slim

WORKDIR /srv

# Install dependencies first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# The highlight pool the bot offers when someone says something nice
# rather than asking a question. Without it that path silently does
# nothing — which is exactly what happened: compliments got silence in
# production while working locally, because data/ was never copied.
# Only this file: episodes.json is 7MB and nothing at runtime reads it,
# since retrieval goes to Pinecone.
COPY data/highlights.json ./data/highlights.json
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
