"""Every decision the bot makes BEFORE it searches anything.

    .venv/bin/python scripts/check_reply_gates.py

Retrieval quality is checked elsewhere (hundred_questions, verify_replies).
This checks the thing those cannot: whether the bot should have opened its
mouth at all. Every wrong reply this account has posted was a gate
decision, not a retrieval one -- the passage was usually correct and the
question was never asked.

Costs nothing. The index is a stub, so no embedding, no Pinecone, no
model call; the only thing exercised is the decision path.

Exits non-zero on any mismatch, so it can gate a deploy.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.x_api import Mention  # noqa: E402
from app.x_bot import MentionBot  # noqa: E402

ANSWER, QUIET = "answer", "quiet"


class StubHit:
    """The three fields format_reply reads. Mirrors tests' FakeHit."""

    title = "LIVE W/ LUCA NETZ & GPT-LIVE: Market Bubble Ep 10"
    timestamp = "3:52:34"
    deep_link = "https://x.com/MarketBubble/status/2075316750439338088"


class StubIndex:
    def __init__(self):
        self.asked: list[str] = []

    async def search(self, query, top_k=None, instruction=None):
        self.asked.append(query)

        class R:
            answer = "Around 3:52:34 he said it."
            hits = [StubHit()]
            refused = False
        return R()


class StubClient:
    bot_user_id = "bot"

    def __init__(self, root_text="", batches=None):
        self.posted: list = []
        self.spent_usd = 0.0
        self._root_text = root_text
        self._batches = list(batches or [])

    async def mentions(self, since_id=None, limit=20):
        return self._batches.pop(0) if self._batches else []

    async def post_by_id(self, post_id):
        return {"id": str(post_id), "text": self._root_text}

    async def replied_to(self, limit=100):
        return set()

    async def reply(self, text, to_post_id, allow_link=False):
        self.posted.append((to_post_id, text))
        return "reply-1"


class StubSummaries:
    def __init__(self):
        self.rows = [{
            "episode_id": "x-999", "title": "Ep 19 — Hunter Biden",
            "summary": "TL;DR — a meme coin that collapsed.",
            "published_at": "2026-09-10T00:00:00Z",
            "url": "https://x.com/MarketBubble/status/1"}]

    async def list_all(self):
        return self.rows


# (label, mention text, root post text, expected)
#
# Every QUIET case below is something that actually happened or is one
# character away from something that did. The comments say which.
CASES: list[tuple[str, str, str, str]] = [
    # --- named while talking to somebody else (16 Sep) ---------------
    ("describes the tool",
     "@vibhu That's why I made @mbubbleSearch because I was having too "
     "much fun 😍 And it is kinda impressive ngl I cooked", "", QUIET),
    ("recommends the tool",
     "@ImPushingSOL soon you gotta add @mbubbleSearch in that\n"
     "cooking something up for the Ansem army 👀", "", QUIET),
    ("names it in praise",
     "@someone honestly @mbubbleSearch is the coolest thing built "
     "this month and nobody is talking about it", "", QUIET),

    # --- but a request that names it mid-sentence is still a request --
    ("introduce yourself",
     "@Clive_99 Yoo @mbubbleSearch introduce yourself", "", ANSWER),
    ("recommended AND asked",
     "@OnlyLJC you gonna love this @mbubbleSearch what did ansem say "
     "about zcash", "", ANSWER),

    # --- the ordinary question -------------------------------------
    ("plain question", "@mbubbleSearch what did ansem say about zcash",
     "", ANSWER),
    ("question with handles first",
     "@Lexx_eth @Kaiz_294 @mbubbleSearch what did banks say about ai",
     "", ANSWER),

    # --- contentless question under a clip (the Kaiz thread) --------
    ("what are they talking about",
     "@Kaiz_294 @mbubbleSearch what are they talking about",
     "Ansem and Banks land on one of the most important edges in "
     "trading: change your mind when the information changes “The "
     "insiders benefited a lot from the launch no matter how they say "
     "they structured the token supply”", ANSWER),

    # --- fixed answers that must never reach retrieval --------------
    ("contract address", "@mbubbleSearch ca?", "", ANSWER),
    ("are you a bot", "@mbubbleSearch are you a bot?", "", ANSWER),
    ("what is this", "@mbubbleSearch what is this", "", ANSWER),
    ("token mechanics",
     "@mbubbleSearch what percent of fees buy back $MBS", "", ANSWER),
    ("holder rewards",
     "@mbubbleSearch how do holder rewards work", "", ANSWER),

    # --- things that are not questions ------------------------------
    ("bare tag", "@mbubbleSearch", "", QUIET),
    ("bare ticker", "@mbubbleSearch zcash", "", QUIET),

    # Praise and "gm" get a highlight, not silence, and that is
    # deliberate: the length check used to swallow them before the
    # highlight path was reached, and an account that ignores its own
    # well-wishers reads as broken. Answer here means "said something",
    # not "searched the archive".
    ("praise only", "@mbubbleSearch this is sick 🔥", "", ANSWER),
    ("gm", "@mbubbleSearch gm", "", ANSWER),

    # --- somebody else's assistant ----------------------------------
    # Handle last, so the leading-handle strip leaves ours in the body:
    # they tagged us too, and "this chart" is not ours to explain.
    ("asked grok, we are in the body",
     "@grok explain this chart @mbubbleSearch", "", ANSWER),
    ("asked grok only",
     "@grok @mbubbleSearch explain this chart", "", QUIET),

    # --- safety -----------------------------------------------------
    ("somebody else's address",
     "@mbubbleSearch whats ansem's contract address", "", ANSWER),
    ("vote bait",
     "@mbubbleSearch what did ansem say about zcash? vote to list MBS "
     "on binance https://t.co/x", "", QUIET),
]


async def run() -> int:
    rows = []
    for label, text, root, expected in CASES:
        # Through tick(), not compose(). Bait, another assistant's
        # question and opt-outs are refused in tick, one level above
        # compose -- a bank that called compose directly reported the
        # bot answering a scam it actually drops, because the harness
        # skipped the layer doing the work.
        real = Mention(id="1", text=text, author_id="a",
                       conversation_id="c")
        warmup = Mention(id="0", text="@mbubbleSearch hello",
                         author_id="z", conversation_id="z")
        client = StubClient(root, batches=[[warmup], [real]])
        index = StubIndex()
        state = Path("/tmp") / f"gate-{abs(hash(label))}.json"
        state.unlink(missing_ok=True)
        bot = MentionBot(
            client, index, summaries=StubSummaries(), state_path=state,
            contract_address="8VjFid8BVGcTPpUzf4PAWsA5nHJ5h2GQNXPEj",
            token_label="$MBS", site="search.lexthedev.com")
        await bot.tick("2026-09-16")        # cold start answers nothing
        await bot.tick("2026-09-16")
        got = ANSWER if client.posted else QUIET
        sent = client.posted[0][1] if client.posted else ""
        rows.append((got == expected, label, expected, got, index.asked,
                     sent[:60]))

    bad = [r for r in rows if not r[0]]
    print(f"\n  {'':<3}{'case':<26}{'want':<8}{'got':<8}reply / searched")
    print("  " + "-" * 92)
    for ok, label, want, got, asked, preview in rows:
        mark = "ok " if ok else "XX "
        detail = preview if got == ANSWER else (
            f"searched {asked!r}" if asked else "")
        print(f"  {mark}{label:<26}{want:<8}{got:<8}{' '.join(detail.split())[:44]}")

    print()
    if bad:
        print(f"  {len(bad)} of {len(rows)} wrong:")
        for _ok, label, want, got, _a, _p in bad:
            print(f"    {label}: wanted {want}, got {got}")
        return 1
    print(f"  all {len(rows)} gate decisions correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
