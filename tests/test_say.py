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


@pytest.mark.parametrize("name", ["about", "automated", "ca"])
def test_every_canned_message_has_something_to_say(name):
    """They are built from the bot's own functions, so a change to the
    description cannot leave this posting a stale version of it."""
    from app.config import get_settings

    say = _say()
    text = say.canned(name, get_settings())
    assert text and len(text) > 40, name
