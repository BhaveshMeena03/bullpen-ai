"""Bullpen Concierge — Discord bot.

An /ask slash command that answers support questions from Bullpen's official
documentation, inside the server where people already ask them.

A separate bot from bot.py rather than a second command on it, because the two
serve different audiences: podcast search is for listeners, this is for users
of the platform. Keeping them apart means a server can install one without the
other, and the rate limiting, HTTP retry behaviour and output safety are
shared through limits.py / format.py instead of being duplicated.

Production concerns handled here:
- Deferred responses (the backend call takes longer than Discord's 3s window).
- Per-user cooldown and a bot-wide ceiling (both guard model spend).
- Question validation and length caps.
- Every failure path returns a friendly, ephemeral error — never a traceback.
- Answers can never post a link or ping a role.

Env:
    DISCORD_TOKEN    (required)  bot token (its own, not the search bot's)
    BACKEND_URL      (required)  e.g. https://marketbubble-search.onrender.com
    GUILD_ID         (optional)  dev guild id for instant command sync
    COOLDOWN_SECONDS (optional)  per-user cooldown, default 8
    CHAT_TIMEOUT     (optional)  backend call timeout seconds, default 60
    MAX_ASKS_PER_MIN (optional)  bot-wide ceiling, default 20
    ALWAYS_EPHEMERAL (optional)  "1" forces every answer private, default off
"""

from __future__ import annotations

import logging
import os
import signal

import discord
from discord import app_commands

from .concierge_client import ChatClient, ChatError
from .concierge_format import build_chat_payload
from .limits import Cooldown, GlobalThrottle
from .memory import ConversationMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("bpcbot")

GREEN = 0x16C784
FOOTER = "Bullpen Concierge · from the docs · not financial advice"

# Shown when someone asks the one question a support bot must never answer.
SEED_PHRASE_WARNING = (
    "⚠️ Never share your seed phrase or private key — with me, with support, "
    "or with anyone who DMs you. No one from Bullpen will ever ask for it."
)
_SECRET_WORDS = ("seed phrase", "seedphrase", "private key", "recovery phrase",
                 "mnemonic", "secret phrase")


# Offered as you type in the question box. An empty prompt tells a new user
# nothing about what the bot knows, so they either guess or don't use it.
# These are the questions a support desk actually gets, so the suggestions
# double as documentation of what it can answer.
COMMON_QUESTIONS = [
    "why is my deposit not showing up?",
    "how do i fund my account?",
    "how do i set a stop loss?",
    "what's the difference between spot and perps?",
    "how do i claim $ANSEM?",
    "what order types can i use?",
    "how do withdrawals work?",
    "why did my order not fill?",
    "how does leverage work?",
    "what is a prediction market?",
    "how do i connect a wallet?",
    "what fees does bullpen charge?",
]
_MAX_SUGGESTIONS = 25  # Discord's hard limit on autocomplete choices


async def suggest_questions(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """Filter the common questions by what has been typed so far.

    Substring rather than prefix: someone typing "deposit" should still be
    offered "why is my deposit not showing up?". Free text always wins — this
    only suggests, it never restricts what can be asked.
    """
    typed = (current or "").strip().lower()
    matches = [q for q in COMMON_QUESTIONS if typed in q] if typed else COMMON_QUESTIONS
    return [
        app_commands.Choice(name=q, value=q)
        for q in matches[:_MAX_SUGGESTIONS]
    ]


def _env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.environ.get(name, default)
    if required and not val:
        raise SystemExit(f"Missing required env var: {name}")
    return val.strip() if isinstance(val, str) else val


class ConciergeBot(discord.Client):
    def __init__(self, *, backend_url: str, timeout: float, guild_id: int | None):
        super().__init__(intents=discord.Intents.none())
        self.tree = app_commands.CommandTree(self)
        self._chat = ChatClient(backend_url, timeout=timeout)
        self._guild_id = guild_id

    async def setup_hook(self) -> None:
        # Guild sync is only a convenience: it registers commands instantly
        # instead of waiting out the ~1h global propagation. So its failure
        # must not be fatal. It used to be, and a GUILD_ID for a server the
        # bot had not been invited to raised 403 here, killed the process, and
        # left Fly restarting it forever — the bot was completely unusable
        # because of an optional dev setting.
        if self._guild_id:
            try:
                guild = discord.Object(id=self._guild_id)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                logger.info("Synced commands to guild %s (instant)", self._guild_id)
                return
            except discord.HTTPException as exc:
                logger.error(
                    "Guild sync to %s failed (%s) — is the bot invited to that "
                    "server, with the applications.commands scope? Falling back "
                    "to a global sync.", self._guild_id, exc,
                )
        await self.tree.sync()
        logger.info("Synced commands globally (propagates within ~1h)")

    async def on_ready(self) -> None:
        logger.info("Logged in as %s", self.user)

    async def close(self) -> None:
        await self._chat.aclose()
        await super().close()

    async def run_ask(
        self,
        question: str,
        history: list[dict] | None = None,
        brief: bool = True,
    ) -> tuple[discord.Embed, str]:
        """Returns the embed and the raw answer, so the caller can remember it
        without re-deriving it from the formatted output."""
        result = await self._chat.ask(question, history=history, brief=brief)
        payload = build_chat_payload(question, result)

        embed = discord.Embed(
            title=payload["title"],
            description=payload["description"],
            colour=GREEN,
        )
        if payload["sources"]:
            embed.add_field(name="from", value=payload["sources"], inline=False)
        embed.set_footer(text=FOOTER)
        # Remember the model's own answer, not the truncated/flattened one the
        # embed shows, so a follow-up sees what it actually said.
        return embed, ("" if payload["empty"] else result.answer)


def build_bot() -> ConciergeBot:
    token = _env("DISCORD_TOKEN", required=True)
    backend = _env("BACKEND_URL", required=True)
    guild_id = _env("GUILD_ID")
    cooldown_s = float(_env("COOLDOWN_SECONDS", "8"))
    timeout = float(_env("CHAT_TIMEOUT", "60"))
    max_per_min = float(_env("MAX_ASKS_PER_MIN", "20"))
    always_ephemeral = _env("ALWAYS_EPHEMERAL", "0") == "1"

    bot = ConciergeBot(
        backend_url=backend,
        timeout=timeout,
        guild_id=int(guild_id) if guild_id else None,
    )
    cooldown = Cooldown(cooldown_s)
    throttle = GlobalThrottle(max_per_min)
    memory = ConversationMemory()
    # Never let a model-generated answer @everyone/@here/@role the channel.
    no_mentions = discord.AllowedMentions.none()
    bot._token = token  # stashed for run()

    @bot.tree.command(
        name="ask",
        description="Ask anything about Bullpen — answered from the official docs.",
    )
    @app_commands.describe(
        question="Type your question, or pick a suggestion",
    )
    @app_commands.autocomplete(question=suggest_questions)
    async def ask_cmd(
        interaction: discord.Interaction,
        question: str,
    ) -> None:
        import time
        now = time.monotonic()

        # One field, no options: every boolean in a slash command renders as a
        # True/False picker, so each one is a step between wanting an answer
        # and getting it. Visibility is a server-wide decision instead, made
        # once by whoever runs the server rather than per question by someone
        # who just wants help. Public by default, since one answer in a busy
        # server serves everyone reading it.
        hidden = always_ephemeral

        # Validate BEFORE consuming any limiter slot.
        question = " ".join(question.split())
        if not (2 <= len(question) <= 300):
            await interaction.response.send_message(
                "Ask a question between 2 and 300 characters.", ephemeral=True
            )
            return

        # Answer this one locally and ephemerally, before spending a model call
        # or writing anything to a public channel. Someone typing their seed
        # phrase into a Discord command needs a warning, not a documentation
        # lookup — and the text must not be echoed back into the channel.
        if any(w in question.lower() for w in _SECRET_WORDS):
            await interaction.response.send_message(
                SEED_PHRASE_WARNING, ephemeral=True, allowed_mentions=no_mentions
            )
            return

        wait = cooldown.retry_after(interaction.user.id, now)
        if wait > 0:
            await interaction.response.send_message(
                f"⏳ one sec — try again in {wait:.0f}s.", ephemeral=True
            )
            return
        if not throttle.allow(now):
            await interaction.response.send_message(
                "🌊 The assistant is at capacity right now — try again shortly.",
                ephemeral=True,
            )
            return

        cooldown.stamp(interaction.user.id, now)
        # MUST defer: the backend call takes longer than Discord's 3s window.
        # Visibility is fixed at defer time — the followup inherits it, so it
        # cannot be decided later once the answer is back.
        await interaction.response.defer(thinking=True, ephemeral=hidden)
        try:
            channel_id = interaction.channel_id or 0
            past = memory.get(interaction.user.id, channel_id, now)
            embed, answer = await bot.run_ask(question, past)
            await interaction.followup.send(
                embed=embed, ephemeral=hidden, allowed_mentions=no_mentions
            )
            # Only remember exchanges that produced a real answer. Storing a
            # refusal would make the model keep apologising for it.
            if answer:
                memory.append(
                    interaction.user.id, channel_id, question, answer, now
                )
        except ChatError as exc:
            await interaction.followup.send(
                f"⚠️ {exc}", ephemeral=True, allowed_mentions=no_mentions
            )
        except Exception:  # noqa: BLE001 — never crash a command
            logger.exception("Unhandled error in /ask")
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
