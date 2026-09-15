"""The one thing that matters about a script that posts in public.

It must not post unless it was told to. Everything else it does is
printing, and printing is recoverable.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAY = ROOT / "scripts" / "say.py"


def _say():
    spec = importlib.util.spec_from_file_location("say", SAY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nothing_is_posted_without_yes():
    """Run for real, with credentials present, and assert it stayed quiet.

    Mocking the client would test the mock. This runs the actual script
    and checks the process said so.
    """
    result = subprocess.run(
        [sys.executable, str(SAY), "automated"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    assert "Nothing was posted" in result.stdout
    assert "posted: https" not in result.stdout


def test_it_refuses_an_ambiguous_instruction():
    """A name and --text together is two different posts, so neither."""
    result = subprocess.run(
        [sys.executable, str(SAY), "about", "--text", "something else"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "exactly one" in (result.stderr + result.stdout)


def test_a_post_url_is_accepted_where_an_id_is_expected():
    say = _say()
    assert say._post_id("2092822509368607177") == "2092822509368607177"
    assert say._post_id(
        "https://x.com/Lexx_eth/status/2092822509368607177"
    ) == "2092822509368607177"
    assert say._post_id(
        "https://x.com/Lexx_eth/status/2092822509368607177?s=20"
    ) == "2092822509368607177"


class _Settings:
    """Everything canned() reads, fixed.

    Stubbed rather than taken from get_settings() because the result
    otherwise depends on whichever .env happens to be loaded. Reading the
    real settings made the "ca" case pass on a machine carrying a contract
    address and fail on every machine without one -- which is exactly how
    it failed the first time CI got far enough to run the suite.
    """

    x_bot_site = "search.lexthedev.com"
    x_bot_contract_address = "8VjFid8BVGcTPpUzf4PAWsA5nHJ5h2GQNXPEj"
    x_bot_token_label = "$MBS"


@pytest.mark.parametrize("name", ["about", "automated", "ca"])
def test_every_canned_message_has_something_to_say(name):
    """They are built from the bot's own functions, so a change to the
    description cannot leave this posting a stale version of it."""
    say = _say()
    text = say.canned(name, _Settings())
    assert text and len(text) > 40, name


def test_the_ca_message_is_absent_rather_than_wrong_without_an_address():
    """No address configured means no message, not an invented one.

    pinned_answer returns None when the address is unset, and say's main()
    prints "nothing to say" and exits 2 on that. Posting a made-up
    44-character string under a token account is the one failure here that
    could cost a reader money, so the absence is the behaviour worth
    pinning down.
    """
    say = _say()

    class _NoAddress(_Settings):
        x_bot_contract_address = None

    assert say.canned("ca", _NoAddress()) is None
