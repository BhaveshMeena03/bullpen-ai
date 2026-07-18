# Market Bubble Search — Discord bot

A `/search` slash command that lets a Discord community search every Market
Bubble episode and jump to the exact YouTube moment, without leaving the
server. It's a **thin client** — it calls the deployed search backend over
HTTP, so it holds only a Discord token (no Anthropic/Voyage/Pinecone keys).

```
Discord  ──/search──▶  bot  ──HTTPS──▶  marketbubble-search.onrender.com
                        │                        │
                    embed reply  ◀── answer + timestamp links
```

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
