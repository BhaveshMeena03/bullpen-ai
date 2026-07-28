# Minimal image for the Discord bot — it's a self-contained package that only
# needs discord.py + httpx (no backend code, no model keys).
FROM python:3.12-slim

WORKDIR /app

COPY discord_bot/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY discord_bot/ ./discord_bot/

# Non-root runtime.
RUN useradd --create-home appuser
USER appuser

# Worker process — connects out to Discord + the backend, serves no HTTP.
#
# One image, two bots. They share the HTTP client, rate limiting and output
# safety, and differ only in entrypoint, so which one runs is a deploy-time
# choice rather than a second image to keep in sync:
#
#   BOT_MODULE=discord_bot.bot            podcast search  (/search)  [default]
#   BOT_MODULE=discord_bot.concierge_bot  support docs    (/ask)
#
# Each deployment needs its own DISCORD_TOKEN — one token is one bot identity.
CMD ["sh", "-c", "python -m ${BOT_MODULE:-discord_bot.bot}"]
