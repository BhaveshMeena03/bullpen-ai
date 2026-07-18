"""Market Bubble Search — Discord bot.

A /search slash command that queries the deployed backend and returns the
answer plus clickable YouTube timestamp links, right inside Discord.

Production concerns handled here:
- Deferred responses (search takes >3s; Discord kills a non-deferred
  interaction at 3s).
- Per-user cooldown (protects the backend's model budget from one spammer).
- Query validation and length caps.
- Every failure path returns a friendly, ephemeral error — the bot never
  shows a raw traceback and never crashes on a single command.
- Clean startup/shutdown of the shared HTTP client.
- Command sync scoped to a guild when GUILD_ID is set (instant) vs global
  (up to 1h to propagate).

Env:
    DISCORD_TOKEN   (required)  bot token
    BACKEND_URL     (required)  e.g. https://marketbubble-search.onrender.com
    GUILD_ID        (optional)  dev guild id for instant command sync
    COOLDOWN_SECONDS(optional)  per-user cooldown, default 8
    SEARCH_TIMEOUT  (optional)  backend call timeout seconds, default 60
"""

from __future__ import annotations

import logging
import os
import signal

import discord
from discord import app_commands

from .client import SearchClient, SearchError
from .format import build_answer_payload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("mbbot")

GREEN = 0x16C784
FOOTER = "Market Bubble Search · not financial advice"


def _env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.environ.get(name, default)
    if required and not val:
        raise SystemExit(f"Missing required env var: {name}")
    return val.strip() if isinstance(val, str) else val


class MBBot(discord.Client):
    def __init__(self, *, backend_url: str, timeout: float, guild_id: int | None):
        super().__init__(intents=discord.Intents.none())
        self.tree = app_commands.CommandTree(self)
        self._search = SearchClient(backend_url, timeout=timeout)
        self._guild_id = guild_id

    async def setup_hook(self) -> None:
        if self._guild_id:
            guild = discord.Object(id=self._guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info("Synced commands to guild %s (instant)", self._guild_id)
        else:
            await self.tree.sync()
            logger.info("Synced commands globally (propagates within ~1h)")

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (%s)", self.user, getattr(self.user, "id", "?"))
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening, name="/search the pod"
            )
        )

    async def close(self) -> None:
        await self._search.aclose()
        await super().close()

    async def run_search(self, query: str) -> discord.Embed:
        """Query the backend and build an embed. Raises SearchError for the
        caller to turn into a friendly message."""
        result = await self._search.search(query)
        payload = build_answer_payload(query, result)
        embed = discord.Embed(
            description=payload["description"],
            color=GREEN if not payload["empty"] else 0x8A939E,
        )
        if payload["title"]:
            embed.set_author(name=f"🔎  {payload['title']}")
        if payload["hits"]:
            embed.add_field(
                name="Jump to the moment", value=payload["hits"], inline=False
            )
        embed.set_footer(text=FOOTER)
        return embed


# --- rate limiting ----------------------------------------------------------
class Cooldown:
    """Per-user cooldown. Discord has its own limits, but this specifically
    guards the backend's model spend from a single user spamming /search."""

    def __init__(self, seconds: float):
        self._seconds = seconds
        self._last: dict[int, float] = {}

    def retry_after(self, user_id: int, now: float) -> float:
        last = self._last.get(user_id, 0.0)
        remaining = self._seconds - (now - last)
        return max(0.0, remaining)

    def stamp(self, user_id: int, now: float) -> None:
        self._last[user_id] = now
        if len(self._last) > 10_000:  # bounded
            oldest = min(self._last, key=self._last.get)
            del self._last[oldest]


class GlobalThrottle:
    """A hard ceiling on searches/minute across the ENTIRE bot — the budget
    backstop. Per-user cooldowns don't stop a coordinated spam raid in a big
    server; this caps total model spend regardless of how many users pile on.
    Token bucket: `per_min` tokens, refilled continuously."""

    def __init__(self, per_min: float):
        self._rate = per_min / 60.0
        self._burst = max(1.0, per_min)
        self._tokens = self._burst
        self._last = 0.0

    def allow(self, now: float) -> bool:
        if self._last == 0.0:
            self._last = now
        self._tokens = min(self._burst, self._tokens + (now - self._last) * self._rate)
        self._last = now
        if self._tokens < 1.0:
            return False
        self._tokens -= 1.0
        return True


def build_bot() -> MBBot:
    token = _env("DISCORD_TOKEN", required=True)
    backend = _env("BACKEND_URL", required=True)
    guild_id = _env("GUILD_ID")
    cooldown_s = float(_env("COOLDOWN_SECONDS", "8"))
    timeout = float(_env("SEARCH_TIMEOUT", "60"))
    max_per_min = float(_env("MAX_SEARCHES_PER_MIN", "20"))

    bot = MBBot(
        backend_url=backend,
        timeout=timeout,
        guild_id=int(guild_id) if guild_id else None,
    )
    cooldown = Cooldown(cooldown_s)
    throttle = GlobalThrottle(max_per_min)
    # Only reply to mentions/roles the model can't manufacture — never let a
    # model-generated answer @everyone/@here/@role the channel.
    no_mentions = discord.AllowedMentions.none()
    bot._token = token  # stashed for run()

    @bot.tree.command(
        name="search",
        description="Search every Market Bubble episode and jump to the moment.",
    )
    @app_commands.describe(question="What do you want to know?")
    async def search_cmd(
        interaction: discord.Interaction, question: str
    ) -> None:
        import time
        now = time.monotonic()

        # Validate BEFORE consuming any limiter slot.
        question = " ".join(question.split())  # collapse whitespace/control chars
        if not (2 <= len(question) <= 300):
            await interaction.response.send_message(
                "Ask a question between 2 and 300 characters.", ephemeral=True
            )
            return

        wait = cooldown.retry_after(interaction.user.id, now)
        if wait > 0:
            await interaction.response.send_message(
                f"⏳ one sec — try again in {wait:.0f}s.", ephemeral=True
            )
            return
        # Bot-wide budget backstop: a coordinated raid can't blow past this.
        if not throttle.allow(now):
            await interaction.response.send_message(
                "🌊 The bot is at capacity right now — try again shortly.",
                ephemeral=True,
            )
            return

        cooldown.stamp(interaction.user.id, now)
        # MUST defer: the backend call takes longer than Discord's 3s window.
        await interaction.response.defer(thinking=True)
        try:
            embed = await bot.run_search(question)
            await interaction.followup.send(embed=embed, allowed_mentions=no_mentions)
        except SearchError as exc:
            await interaction.followup.send(
                f"⚠️ {exc.message}", ephemeral=True, allowed_mentions=no_mentions
            )
        except Exception:  # noqa: BLE001 — never crash a command
            logger.exception("Unhandled error in /search")
            await interaction.followup.send(
                "⚠️ Something went wrong — please try again.",
                ephemeral=True, allowed_mentions=no_mentions,
            )

    @bot.tree.error
    async def on_app_error(
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        logger.exception("App command error: %s", error)
        msg = "⚠️ Something went wrong — please try again."
        try:
            if interaction.response.is_done():
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=True)
        except discord.HTTPException:
            pass

    return bot


def main() -> None:
    bot = build_bot()

    # Graceful shutdown on SIGTERM (containers/hosts send this).
    def _stop(*_):
        logger.info("Shutting down…")
        import asyncio
        asyncio.get_event_loop().create_task(bot.close())

    try:
        signal.signal(signal.SIGTERM, _stop)
    except ValueError:
        pass  # not in main thread (e.g. under a supervisor)

    bot.run(bot._token, log_handler=None)  # we configure logging ourselves


if __name__ == "__main__":
    main()
