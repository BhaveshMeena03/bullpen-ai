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
   GUILD_ID=your-server-id     # optional: instant command sync in dev
   COOLDOWN_SECONDS=8          # optional
   SEARCH_TIMEOUT=60           # optional
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

## Security notes
- The bot needs **only** the Discord token — it can't touch your model or
  vector-DB keys, because those live on the backend.
- Rotate the Discord token if it ever leaks (Developer Portal → Reset Token).
