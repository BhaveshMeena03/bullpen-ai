"""The reply checker has to be able to fail, too.

verify_replies reported zero problems on its first run. That is either
good news or a checker that cannot detect anything, and the difference
matters more than the number. Every case here is a real defect from this
week, fed in directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_replies import CITED, LINK, T_PARAM, seconds, supported  # noqa: E402

# One real passage, standing in for a transcript window.
TRANSCRIPT = (
    "what trump is saying is we're going to make hyperliquid legal in the "
    "u.s. so that the same amount of people who can easily use robinhood, "
    "coinbase, all these things can also easily use hyperliquid"
)


def test_a_link_that_jumps_somewhere_else_is_caught():
    """The reply said "scheduled to appear around 3:30 PM" and offered a
    jump to 3 minutes 30 seconds — 210 seconds into an unrelated show."""
    reply = "Jump to 27:51:\nhttps://www.youtube.com/watch?v=abc&t=210s"
    cited = CITED.findall(reply)[0]
    param = int(T_PARAM.search(LINK.search(reply).group(0)).group(1))
    assert seconds(cited) == 1671
    assert abs(param - seconds(cited)) > 3, (
        "checker cannot see that the link and the citation disagree")


def test_a_link_that_matches_passes():
    reply = "Jump to 27:51:\nhttps://www.youtube.com/watch?v=abc&t=1671s"
    cited = CITED.findall(reply)[0]
    param = int(T_PARAM.search(LINK.search(reply).group(0)).group(1))
    assert abs(param - seconds(cited)) <= 3


def test_a_clock_time_is_not_read_as_a_citation():
    """"scheduled to appear around 3:30 PM" must not become a moment."""
    assert not CITED.findall("he was scheduled to appear around 3:30 PM")
    assert CITED.findall("around 3:30 he said it") == ["3:30"]


def test_hours_minutes_and_seconds_all_convert():
    assert seconds("27:51") == 1671
    assert seconds("2:28:13") == 8893
    assert seconds("0:00") == 0


def test_an_unsupported_claim_is_caught():
    """The transcript at that second is about nuclear proliferation; the
    answer claims Solana at $150. Nothing in common."""
    claim = "Solana is expected to reach 150 during Q3"
    assert not supported("cold war nukes bunkers tech elites", claim)


def test_a_supported_claim_passes():
    claim = 'Trump wants to make Hyperliquid legal in the U.S. so Robinhood '
    assert supported(TRANSCRIPT, claim)


def test_a_number_in_the_claim_is_enough():
    """A digit shared with the transcript is enough. "1 billion" against
    "a billion" is not a shared digit — the check is deliberately literal,
    because a looser one would call any claim supported."""
    assert supported("they made 1.2 million the day before", "made 1.2 million")
    assert not supported("they made a billion dollars", "made 1 billion")


def test_the_link_pattern_survives_a_trailing_colon():
    reply = "Full episode:\nhttps://x.com/MarketBubble/status/2093074127011909791"
    assert LINK.search(reply).group(0).endswith("2093074127011909791")
