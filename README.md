# Bullpen Concierge

AI support agent + podcast search for the BullpenFi trading terminal
(Solana spot, Hyperliquid perps, Polymarket prediction markets, and the
$ANSEM airdrop claim flow). Async FastAPI backend, RAG over Pinecone,
Claude for reasoning, and an embeddable one-script-tag chat widget.

## Project layout

```
app/                    backend package
  main.py               API routes, CORS, static mounts
  agent.py              Claude integration, guardrail system prompt, caching
  retriever.py          query embedding + Pinecone similarity search
  ingest.py             chunkers (docs / transcripts / tweets) + upsert
  podcast.py            Market Bubble episode search (timestamped RAG)
  security.py           per-IP rate limiting + admin-token auth
  config.py             pydantic-settings (all secrets via env / .env)
  schemas.py            request/response models
widget/                 embeddable chat widget (vanilla JS, no build step)
demo/                   demo terminal page, podcast search page, mock server
scripts/                redteam.py, fetch_episodes.py, ingest_episodes.py, GIF makers
tests/                  61 offline tests (no API keys needed)
data/                   seed knowledge + fetched episode transcripts
.github/workflows/      CI: ruff + pytest on every push
Dockerfile              non-root container with healthcheck
```

## Architecture

```
client ──> FastAPI (main.py) ── rate limit / admin auth (security.py)
              │
              ├── Retriever (retriever.py) ── Voyage AI embed ──> Pinecone (metadata pre-filter)
              │
              ├── PodcastIndex (podcast.py) ── namespace "podcast" ──> timestamp deep-links
              │
              └── ConciergeAgent (agent.py) ──> Anthropic API
                       model-aware request builder (Opus 4.8 default)
                       └─ Fable 5 mode: server-side refusal fallback

ingest.py: docs / transcripts / tweets ──> chunk ──> embed ──> Pinecone
```

## Key model-integration decisions

- **Model**: `claude-opus-4-8` by default (strong quality at $5/$25 per
  MTok). Swap via `ANTHROPIC_MODEL` — `claude-sonnet-5` to go cheaper,
  `claude-fable-5` for max capability. The agent adapts the request
  shape automatically per model.
- **Thinking**: on Opus 4.8 / Sonnet 5 the agent requests
  `thinking: {type: "adaptive"}` explicitly — this is what makes the
  model verify against retrieved context before answering. On Fable 5
  thinking is always on and the parameter must be omitted (an explicit
  config 400s). Depth is controlled with `output_config.effort`
  (`EFFORT` env var, default `high`).
- **No sampling params**: `temperature`/`top_p`/`top_k` are rejected on
  these models; behavior is steered via the system prompt.
- **Prompt caching**: frozen system prompt carries a `cache_control`
  breakpoint; a second breakpoint sits on the newest user turn so
  multi-turn conversations replay the prior prefix at ~0.1x input price.
  Verify with `usage.cache_read_input_tokens` in `/v1/chat` responses.
- **Refusal handling**: safety classifiers can return
  `stop_reason: "refusal"`. On Fable 5 the server-side fallback beta
  (`server-side-fallback-2026-06-01`) re-serves declined requests on
  Opus 4.8 inside the same call. On every model, a refusal returns a
  safe canned support message instead of a crash.

## Guardrails

Enforced in the system prompt (and honored over user/context injection):
no financial advice, no price predictions, never request or accept private
keys / seed phrases, always identifies as a support tool, factual risk
disclosure for perps and prediction markets.

## Widget & demo page

The helpmate "lives on a website" as an embeddable widget — the host site
adds one script tag and the chat bubble appears, talking to this backend:

```html
<script src="https://your-host/widget/bullpen-concierge.js"
        data-api-base="https://your-host" defer></script>
```

- `widget/bullpen-concierge.js` — dependency-free vanilla JS chat bubble
  (streams tokens from `/v1/chat/stream`, renders source chips, replays
  history so prompt caching keeps working).
- `demo/index.html` — a mock of the Bullpen perps terminal
  (app.bullpen.fi/perps/BTC look-alike) with the widget embedded.
- `demo/mock_server.py` — zero-dependency demo server with canned SSE
  answers, so the whole experience runs with **no API keys**:

```bash
python3 demo/mock_server.py   # -> http://localhost:8000/demo/
```

The real backend serves the same paths (`/demo/`, `/widget/...`), so once
your keys are in `.env`, `uvicorn app.main:app` gives you the identical
demo backed by live Claude + RAG answers.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys
uvicorn app.main:app --reload
```

Create the Pinecone index once (dimension must match `EMBEDDING_DIMENSION`,
metric `cosine`), then seed it with the starter knowledge base — who Ansem
and FaZe Banks are, what Bullpen and $ANSEM are, and beginner crypto
concepts (`data/seed.json`):

```bash
curl -X POST localhost:8000/v1/ingest \
  -H 'content-type: application/json' -d @data/seed.json
```

Add your own docs/transcripts/tweets the same way (see `IngestDocument`
in `app/schemas.py` for the shape).

### How question types are answered

The system prompt gives the agent three knowledge tiers:

1. **General crypto education** ("what is a wallet / gas / slippage?") —
   answered from Claude's own knowledge, education-only framing.
2. **Platform & people** ("who is Ansem?", "what is Bullpen / $ANSEM?") —
   answered from stable facts baked into the system prompt plus retrieved
   context.
3. **Operational specifics** (fees, dates, addresses, eligibility
   thresholds) — only ever answered from retrieved context; otherwise the
   agent says it doesn't know and points to official support.

Chat:

```bash
curl -X POST localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"message": "How do I claim the $ANSEM airdrop?", "filters": {"source_type": "docs"}}'

# streaming (SSE)
curl -N -X POST localhost:8000/v1/chat/stream -H 'content-type: application/json' \
  -d '{"message": "What is a seed phrase?"}'
```

## Security & operations

- **Rate limiting** — public endpoints (`/v1/chat*`, `/v1/podcast/search`)
  are token-bucket limited per client IP (`RATE_LIMIT_RPM`, default 30).
  In-memory by design for single-process; swap to Redis for replicas.
- **Admin auth** — set `ADMIN_TOKEN` and the ingestion endpoints require
  it via the `X-Admin-Token` header. Unset = open (local dev only).
- **CORS** — `*` for the demo; lock `allow_origins` to the host site's
  domain before production.
- **Secrets** — env/.env only (pydantic-settings); `.env` is gitignored.
- **CI** — GitHub Actions runs ruff + the full test suite on every push.
- **Docker** — `docker build -t bullpen-concierge . && docker run --env-file .env -p 8000:8000 bullpen-concierge`
  (non-root user, healthcheck on `/healthz`).

## Market Bubble episode search

A second, standalone tool: semantic search over "Market Bubble" podcast
transcripts. Ask a question in plain English, get an answer plus citations
that **deep-link to the exact timestamp** in the episode
(`youtube.com/watch?v=...&t=872s`).

- `app/podcast.py` — timestamped chunking (windows keep their start time),
  a dedicated Pinecone namespace (`podcast`), and the same guardrails as
  the concierge (informational only, never advice, no invented quotes).
- `demo/podcast.html` — the search page.
- Endpoints: `POST /v1/podcast/ingest`, `POST /v1/podcast/search`.

Seed with sample episodes (generic "Host/Co-host" lines — **placeholder,
not real quotes**):

```bash
curl -X POST localhost:8100/v1/podcast/ingest \
  -H 'content-type: application/json' -d @data/episodes.sample.json
```

### Pulling real episodes

`scripts/fetch_episodes.py` pulls real Market Bubble captions from YouTube
and converts them (rolling-overlap dedup, HTML/bleep cleanup, timestamped
segments) into `data/episodes.json`:

```bash
.venv/bin/python scripts/fetch_episodes.py --cookies chrome VIDEO_ID ...
.venv/bin/python scripts/fetch_episodes.py --cookies chrome --latest 5
```

YouTube gates caption downloads, so the pull needs three things (all wired
in): **`--cookies chrome`** (passes the bot check with your logged-in
browser), a **JS runtime** — `brew install deno` — and yt-dlp's
**`--remote-components ejs:github`** solver (added automatically). If a pull
is still blocked, download the `.vtt` by hand (browser extension) and convert
offline — no network needed:

```bash
.venv/bin/python scripts/fetch_episodes.py --vtt episode.vtt=VIDEO_ID
```

Then ingest `data/episodes.json` via `/v1/podcast/ingest`.

> The parser is unit-tested (`tests/test_vtt_parser.py`) against YouTube's
> rolling-window caption format. Auto-captions have no speaker names and
> occasional transcription errors — fine for search, but don't render raw
> auto-caption text as verbatim quotes attributed to a host.

## Testing

```bash
.venv/bin/python -m pytest tests/            # 44 offline tests, no keys needed
ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python scripts/redteam.py   # live red-team
```

- `tests/` — chunking strategies, metadata tagging, the Fable 5 request
  contract (no `thinking` param, no sampling params, fallback beta, cache
  breakpoint placement), API layer with stubbed model (SSE framing,
  refusal path, error mapping, CORS, validation), and demo routing.
- `scripts/redteam.py` — ~25 adversarial probes against the LIVE model
  through the production code path: financial-advice traps, seed-phrase
  phishing, jailbreaks, prompt injection planted in retrieved documents,
  hallucination checks (fees/dates/addresses with no context), identity
  questions, and a prompt-cache verification across turns. Writes
  `redteam_report.md` with the full transcript for human review.
