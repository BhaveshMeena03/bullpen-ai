"""See exactly what the X bot would reply, without touching X.

    .venv/bin/python scripts/preview_reply.py "what did luca netz say about pudgy penguins"
    .venv/bin/python scripts/preview_reply.py --suite

`run_x_bot.py --dry-run` still calls X to read real mentions, which costs
money and needs a credit balance. This calls nothing on X at all: it takes
the text of a post, runs it through the same question parsing, the same
pinned answers and the same reply formatting the live bot uses, and prints
the result with its character count.

That matters because every reply defect found so far came from looking at
composed output on a real question, not from a test:

  - a citation attached to "I couldn't find that", pointing at an unrelated
    episode
  - two different timestamps in one reply, the model's own and the passage
    start, minutes apart
  - Markdown asterisks and [1:39:32] excerpt markers, from a prompt written
    for a page that renders both

None of those fail a test until you know to write it. So look at the output.

Costs the Anthropic call only (~$0.008), or nothing at all for a pinned
answer or a repeat that the answer cache already holds.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.podcast import PodcastIndex  # noqa: E402
from app.x_api import LinkInReplyError, assert_linkless  # noqa: E402
from app.x_bot import (  # noqa: E402
    format_highlight,
    format_reply,
    is_a_pleasantry,
    load_highlights,
    looks_like_a_question,
    pinned_answer,
    question_from,
    weighted_length,
)

# A spread that has caught something before: one answerable from YouTube, one
# only in an X broadcast, a pinned address, a question with no answer, and a
# tag with nothing attached.
SUITE = [
    "@mbubbleSearch what did luca netz say about pudgy penguins",
    "@mbubbleSearch why does ansem think ethereum is done",
    "@mbubbleSearch what did brian armstrong say about the sec",
    "@mbubbleSearch what did chris gilbert say about squire",
    "@mbubbleSearch what's the CA",
    "@mbubbleSearch what did taylor swift say about bitcoin",
    "@mbubbleSearch",
    # Praise, which never reaches retrieval at all — see below.
    "@mbubbleSearch you are so freaking cool",
    "@mbubbleSearch gm king",
]


async def preview(index: PodcastIndex, post: str, settings) -> None:
    question = question_from(post)
    print(f"\n  ── {post}")
    if len(question) < 6:
        print("     (a tag with no question — the bot stays quiet)")
        return

    # Praise is answered from the highlight pool and never touches
    # retrieval. This script did not model that branch, so previewing "you
    # are so freaking cool" printed a retrieval answer — a reply the live
    # bot would never send. A preview tool that is wrong about a whole
    # class of reply is worse than not having one, because the output
    # looks exactly as authoritative as the correct cases beside it.
    if not looks_like_a_question(question) and is_a_pleasantry(question):
        pool = load_highlights()
        if pool:
            import hashlib
            seed = str(abs(hash(post)))
            chosen = pool[int(hashlib.sha256(seed.encode()).hexdigest(), 16)
                          % len(pool)]
            reply = format_highlight(chosen, seed,
                                     settings.x_bot_include_links,
                                     settings.x_bot_post_limit)
            print("     | " + reply.replace("\n", "\n     | "))
            print(f"     {weighted_length(reply)}/"
                  f"{settings.x_bot_post_limit} chars · highlight pool "
                  f"({len(pool)} entries) · no model call · $0")
            return
        print("     (praise, but the highlight pool is empty)")
        return

    pinned = pinned_answer(question, settings.x_bot_contract_address,
                           settings.x_bot_token_label)
    if pinned:
        reply, source = pinned, "pinned · no model call · $0"
    else:
        result = await index.search(question)
        if getattr(result, "refused", False):
            print("     (refused by the model — the bot stays quiet)")
            return
        # The configured limit, not the module default. Omitting this fell
        # back to POST_LIMIT (280) and previewed every answer trimmed to a
        # third of what production actually posts — the live account has
        # replied at 362, 659, 786 and 857 characters, and this printed
        # them cut off at 280 with an ellipsis. A preview that shortens
        # the thing being previewed is not a preview.
        reply = format_reply(result.answer, result.hits,
                             include_links=settings.x_bot_include_links,
                             limit=settings.x_bot_post_limit)
        source = f"{len(result.hits)} hits · {result.model}"

    for line in reply.splitlines():
        print(f"     | {line}")
    # As X counts it, not as Python does: every URL is 23 characters
    # whatever its length. Counting raw put a fine 268-character reply at
    # "295/280 OVER LIMIT" — a preview that talks you out of a reply that
    # would have posted correctly is worse than no preview.
    shown = weighted_length(reply)
    over = " OVER LIMIT" if shown > settings.x_bot_post_limit else ""
    print(f"     {shown}/{settings.x_bot_post_limit} chars · {source}{over}")

    try:
        assert_linkless(reply)
    except LinkInReplyError as exc:
        # Only reachable with links deliberately enabled; worth saying so,
        # because it is the difference between $0.015 and $0.200 a reply.
        print(f"     COSTS $0.200: {exc}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("post", nargs="*", help="the text of a post tagging the bot")
    ap.add_argument("--suite", action="store_true",
                    help="run a fixed spread of questions")
    args = ap.parse_args()

    posts = SUITE if args.suite else [" ".join(args.post)]
    if not posts or not posts[0].strip():
        ap.error("give a post, or --suite")

    settings = get_settings()
    print(f"  reply limit {settings.x_bot_post_limit} chars · links "
          f"{'ON' if settings.x_bot_include_links else 'off'}")
    index = PodcastIndex()
    for post in posts:
        await preview(index, post, settings)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
