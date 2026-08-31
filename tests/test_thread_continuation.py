"""Do not answer a conversation the account is merely standing in.

X carries a participant's handle into every subsequent reply in a
thread. Once this account has answered once, every message between two
other people arrives with @mbubbleSearch in the mention prefix, looking
exactly like a fresh summons.

    "@TheGreatCattsby @mbubbleSearch 🤣😂 it only answers from what was
     said on the broadcast sorry 😅"

That was an aside between two friends. It got "I couldn't find that in
the episodes I've indexed" posted underneath it, in public, on a thread
that had nothing to do with the archive.

An auto-carried mention cannot be distinguished from a typed one -- both
arrive as the handle in the text -- so the tie is broken on the ANSWER
instead. A miss is a good reply to a real question and a bad one to
somebody's joke, so in a thread already answered the miss is swallowed.
A follow-up that actually finds something still posts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.x_bot import (  # noqa: E402
    asks_for_the_latest,
    asks_something,
    is_a_miss,
    question_from,
    salvage,
    summary_request,
)


def would_reply(asked: str) -> bool:
    """The gate compose applies in a thread already answered."""
    return (asks_something(asked) or summary_request(asked) is not None
            or asks_for_the_latest(asked))

ASIDE = ("@TheGreatCattsby @mbubbleSearch 🤣😂 it only answers from what "
         "was said on the broadcast sorry 😅")
MISS = "I couldn't find that in the episodes I've indexed."


class TestTheMessageThatCausedIt:
    def test_the_handle_is_stripped_and_an_aside_remains(self):
        asked = question_from(ASIDE)
        assert "mbubbleSearch" not in asked
        assert "it only answers from what was said" in asked

    def test_the_reply_it_produced_is_a_bare_miss(self):
        """Nothing salvageable, no citation -- the least useful thing the
        account can say."""
        assert is_a_miss(MISS)
        assert salvage(MISS) is None


class TestTheGate:
    source = (ROOT / "app" / "x_bot.py").read_text()

    def test_a_miss_in_an_answered_thread_stays_quiet(self):
        assert "already_here and is_a_miss(result.answer) and not rescued" \
            in self.source

    def test_the_conversation_is_read_from_the_mention(self):
        """`thread` is bound in the poll loop, not in compose(). Using it
        here would have been a NameError on the first miss."""
        assert "conversation = str(mention.conversation_id or mention.id)" \
            in self.source

    def test_it_is_scoped_before_use(self):
        import ast
        tree = ast.parse(self.source)
        compose = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.AsyncFunctionDef)
                       and n.name == "compose")
        body = ast.get_source_segment(self.source, compose)
        assert body.index("conversation = str(") < \
            body.index("conversation_replies.get(conversation")

    def test_only_a_miss_is_swallowed(self):
        """A real answer in a continued thread must still post -- people
        do ask follow-ups, and silencing those would be worse than the
        bug being fixed."""
        gate = self.source[self.source.index("already_here and is_a_miss"):]
        gate = gate[:gate.index("return None") + 11]
        assert "is_a_miss(result.answer)" in gate
        assert "not rescued" in gate

    def test_a_first_reply_in_a_thread_is_unaffected(self):
        """already_here is False when the account has not spoken there, so
        an honest "not in the archive" still reaches somebody who asked."""
        assert "conversation_replies.get(conversation, 0) > 0" in self.source


class TestChatterInAnAnsweredThread:
    """The reply that started this: michael catt wrote "It's 81 jobs to be
    specific", correcting a joke about a producer's workload, and got an
    answer about AI disrupting the labour market. He had not tagged
    anything -- X carried the handle in."""

    @pytest.mark.parametrize("chatter", [
        "It's 81 jobs to be specific",
        "lmao that's crazy",
        "this is sick",
        "yeah exactly",
        "he really does work that hard",
        "😂😂",
    ])
    def test_it_says_nothing(self, chatter):
        assert not would_reply(chatter), chatter

    @pytest.mark.parametrize("asked", [
        "what did ansem say about zcash",
        "what about hyperliquid",
        "summarize episode 17",
        "tldr episode 16",
        "who is michael catt",
        "summarize the latest episode",
        "how much did bonk go to",
    ])
    def test_a_genuine_follow_up_still_lands(self, asked):
        """Silencing these would be worse than the bug -- people do ask
        second questions, and the summary paths do not read as questions
        at all."""
        assert would_reply(asked), asked

    def test_the_gate_runs_before_retrieval(self):
        """A thread this account is only standing in should cost nothing:
        no embedding, no Pinecone query, no rerank, no model call."""
        source = (ROOT / "app" / "x_bot.py").read_text()
        gate = source.index("no question in a thread already answered")
        search = source.index("await self._index.search(")
        assert gate < search
