# Discord bots

Two bots, one package. Both are **thin clients** — they call the deployed
backend over HTTP and hold only a Discord token (no Anthropic/Voyage/Pinecone
keys).

| Bot | Command | Answers from | Module |
|---|---|---|---|
| Market Bubble Search | `/search` | podcast transcripts, links to the exact YouTube second | `discord_bot.bot` |
| Bullpen Concierge | `/ask` | Bullpen's official documentation, cites the pages | `discord_bot.concierge_bot` |

```
Discord  ──/search──▶  bot  ──HTTPS──▶  /v1/podcast/search  ── answer + timestamp links
Discord  ──/ask─────▶  bot  ──HTTPS──▶  /v1/chat            ── answer + doc page names
```

They're separate bots rather than two commands on one, because they serve
different audiences: `/search` is for listeners of the show, `/ask` is for
people using the platform. A server can install either without the other. The
parts worth getting right once — rate limiting (`limits.py`), URL defanging
and truncation (`format.py`), retry behaviour — are shared, not duplicated.

Each bot needs **its own** `DISCORD_TOKEN`; one token is one bot identity.
The image picks which to run from `BOT_MODULE` (see the Dockerfile).

## Who sees the answer

`/ask` is public by default, because in a busy server one answer serves
everyone reading it and cuts the repeat questions the mods handle by hand.

The asker can add `private: true` to keep an answer to themselves. That's
their call to make rather than the bot's: only they know whether the question
gives away their own balance or position. A server that would rather keep
every support answer private sets `ALWAYS_EPHEMERAL=1` and the flag stops
mattering.

Visibility is fixed when the interaction is deferred, so it can't be decided
after the answer comes back.

## Answer length

`/ask` returns a chat-sized answer by default. The full-length one takes about
ten seconds to generate and arrives as a wall of text in Discord, where it goes
unread; the short one takes about four and fits on screen. Same retrieval, same
sources, same guardrails — only the length differs.

`detailed: true` gets the long form for anyone who wants it. The web page has
room to read a thorough answer, so it never asks for brief.

## Follow-up questions

The concierge often asks a clarifying question before answering — "which
account were you funding?" — and then uses the reply. That needs conversation
history, which the API is stateless about: the client replays it, exactly as
the web page does.

`memory.py` keeps that history briefly and forgetfully: keyed on
(user, channel) so nothing bleeds between people or rooms, expiring after ten
minutes of silence, capped at the last few turns since history is resent on
every request, and bounded in total so a busy server can't grow it without
limit. Nothing touches disk — a restart forgets everything, which is the right
default for support chat that may mention someone's balance.

Refused answers are not remembered; storing one makes the model keep
apologising for it.

## The concierge's extra rule

`/ask` intercepts anything mentioning a seed phrase, private key, recovery
phrase or mnemonic **before** it reaches the model, and replies ephemerally
with a warning. Someone typing their seed phrase into a Discord command needs
to be told to stop, not given a documentation lookup — and the text must never
be echoed back into a public channel.

## What's production-grade about it
- **Deferred responses** — search takes >3s; Discord kills a non-deferred
  interaction at 3s, so every command defers immediately then follows up.
- **Per-user cooldown** — protects the backend's model budget from one user
  spamming `/search`.
- **Hardened errors** — timeouts, 5xx (retried once), 503 "busy", 429, and
  any unexpected error all become a friendly ephemeral message. The bot
  never shows a traceback and never crashes on a single command.
- **Input validation** — query length bounded (2–300 chars).
- **Graceful shutdown** — closes the shared HTTP client on SIGTERM.
- **Tested logic** — the client (mocked HTTP), formatting, and cooldown are
  unit-tested (`tests/test_discord_bot.py`); only the Discord gateway glue
  is untestable without a live connection.

## Setup (one-time, ~5 min)

1. **Create the app**: [discord.com/developers/applications](https://discord.com/developers/applications)
   → New Application → name it → **Bot** tab → **Reset Token** → copy the token.
2. **Invite it**: **OAuth2 → URL Generator** → scopes `bot` + `applications.commands`
   → permission `Send Messages` → open the generated URL → add to your server.
   (No privileged intents needed — the bot uses `Intents.none()`.)
3. **Configure** — set env vars (or a `.env`):
   ```
   DISCORD_TOKEN=your-bot-token
   BACKEND_URL=https://marketbubble-search.onrender.com
   GUILD_ID=your-server-id      # optional: instant command sync in dev
   COOLDOWN_SECONDS=8           # optional: per-user cooldown
   MAX_SEARCHES_PER_MIN=20      # optional: bot-wide budget ceiling
   SEARCH_TIMEOUT=60            # optional: backend call timeout
   ```
   (Get GUILD_ID by right-clicking your server with Developer Mode on.)

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r discord_bot/requirements.txt
export $(grep -v '^#' .env | xargs)   # or set env vars however you like
python -m discord_bot.bot
```

- With `GUILD_ID` set, `/search` appears in that server **instantly**.
- Without it, commands sync **globally** and can take up to ~1 hour to show.

## Deploy (always-on)
Any host that runs a long-lived process works — a `$5` VPS, Railway,
Fly.io, or a Render **Background Worker** (not a Web Service — the bot
holds a gateway connection, it doesn't serve HTTP). Set the same env vars
there and run `python -m discord_bot.bot`. On a container host, SIGTERM
triggers a clean shutdown.

## Security

The bot is built to be hard to compromise and impossible to turn into a
weapon even if it were:

**No inbound attack surface.** The bot is outbound-only — it connects *out*
to Discord's gateway and to your backend. It listens on **no port**, so
there is nothing to attack over the network.

**Least privilege.**
- `Intents.none()` — it cannot read messages, member lists, presence, or any
  privileged data. It only receives its own slash-command invocations.
- Invite it with the **single** permission `Send Messages`. Even a fully
  compromised bot could not ban, kick, delete, or manage anything.
- It holds **only** the Discord token — no Anthropic / Voyage / Pinecone
  keys. Those never leave the backend, so the bot can't leak them.

**Abuse / budget protection (defense in depth).**
- **Per-user cooldown** (`COOLDOWN_SECONDS`, default 8s) — stops one user
  spamming `/search`.
- **Bot-wide throttle** (`MAX_SEARCHES_PER_MIN`, default 20) — a hard ceiling
  on total searches/minute, so a coordinated raid in a large server can't
  blow through your model budget.
- **Backend limits** — the API also enforces its own global + per-IP rate
  limits, so the bot is one more layer, not the only one.
- **Input bounds** — questions are whitespace-collapsed and capped at
  2–300 chars before any work happens.

**No mention abuse.** All replies set `AllowedMentions.none()`, so a
model-generated answer can never `@everyone` / `@here` / ping a role.

**No secret leakage.** The token is read from the environment, never logged,
and never echoed to users. Errors return generic messages — never a
traceback or the backend URL. (A test asserts the token isn't logged.)

### The "compromised bot posts a wallet-drainer link" scenario

This is the nightmare in crypto Discords, so it's worth being precise. There
are two distinct paths, with different defenses:

**Path 1 — backend or model manipulation (structurally blocked).** Even if
the search backend were compromised, or a poisoned transcript tricked the
model into producing "launch now, claim $ANSEM 👉 scam.link", the bot
**cannot post that link.** Two hard rules enforce it:
- The bot only ever posts links whose host is a **YouTube domain**
  (`ALLOWED_LINK_HOSTS`). Any other domain in a result is dropped, always.
- The answer text is **URL-stripped** before posting (`defang_urls`), so a
  link can't ride in through the answer body either — and the echoed
  question is stripped too.

  So: whatever the backend returns, the only clickable thing this bot will
  ever post is a `youtube.com` / `youtu.be` link. (Tests enforce this.)

**Path 2 — Discord token theft (must be prevented operationally).** If an
attacker steals the `DISCORD_TOKEN`, they control the bot's identity via
Discord's API directly — they don't run this code, so none of the above
applies. This is true of *every* Discord bot in existence; the defenses are:
- **Never let the token leak.** Store it in your host's secret manager, never
  in a committed file. Rotate it if a laptop/host is ever exposed.
- **Least privilege limits the blast radius.** The bot has only *Send
  Messages* and `Intents.none()`, so even a hijacked token can only post
  messages in channels it can already see — it cannot ban, edit others,
  manage roles, or touch announcement channels it wasn't given.
- **Server-side backstops (recommend to the server admins):** don't grant the
  bot access to announcement/@everyone channels; enable Discord **AutoMod**
  link filtering; the moment anything looks off, **kick the bot** (instantly
  stops it) and reset the token.

**Incident response (if the bot is ever posting bad content):**
1. In the server: **kick/ban the bot** — this stops it immediately, no matter
   what's controlling it.
2. Developer Portal → your app → Bot → **Reset Token** — the stolen token
   dies instantly.
3. Update the token in your host and restart. Re-invite with the same
   least-privilege scope.

**Do not** commit the token or a `.env` containing it. Keep it in your host's
secret manager (Render/Railway/Fly all have one).
