"""The check that would have caught the outage /healthz missed.

A build failed, Render kept serving the previous image, and for twenty
minutes the URL answered, /healthz returned {"status": "ok"} and the logs
were quiet while the bot replied to nobody. Every signal said healthy.

So these pin the two things that were not observable: which build is
actually running, and whether the poll loop is still completing cycles.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.x_bot import Heartbeat  # noqa: E402

WATCHER = ROOT / "scripts" / "watch_x_bot.py"


def _watcher():
    """Import the watcher by path — scripts/ is not a package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("watch_x_bot", WATCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- liveness ---------------------------------------------------------------

def test_a_bot_that_is_switched_off_is_not_broken():
    """Off on purpose must not read the same as dead, or the alarm is
    firing every time the switch is down and nobody looks at it."""
    beat = Heartbeat(started_at=time.time() - 10_000, enabled=False)
    assert beat.healthy()


def test_a_starting_bot_is_given_time_to_finish_its_first_cycle():
    """The first cycle builds clients and reads the account's own timeline
    back from X, so it is slow, and it is not a fault."""
    beat = Heartbeat(started_at=time.time(), enabled=True)
    assert beat.healthy()


def test_a_bot_that_never_polls_is_eventually_called_dead():
    beat = Heartbeat(started_at=time.time() - 600, enabled=True)
    assert not beat.healthy(), "ten minutes with no first poll is dead"


def test_a_polling_bot_is_healthy():
    beat = Heartbeat(started_at=time.time() - 600, enabled=True,
                     poll_seconds=20)
    beat.polled()
    assert beat.healthy()


def test_a_stopped_loop_is_noticed():
    """The failure mode itself: the process alive, the loop not running."""
    beat = Heartbeat(started_at=time.time() - 600, enabled=True,
                     poll_seconds=20)
    beat.polled()
    beat.last_poll_at = time.time() - 300
    assert not beat.healthy()


def test_one_slow_cycle_does_not_raise_an_alarm():
    """A single slow answer must not page anyone — three missed in a row
    is a fault, one cycle running long is a Tuesday."""
    beat = Heartbeat(started_at=time.time() - 600, enabled=True,
                     poll_seconds=20)
    beat.polled()
    beat.last_poll_at = time.time() - 45
    assert beat.healthy()


def test_a_successful_poll_clears_the_error_count():
    beat = Heartbeat(started_at=time.time(), enabled=True)
    beat.failed()
    beat.failed()
    assert beat.consecutive_errors == 2
    beat.polled()
    assert beat.consecutive_errors == 0


def test_the_report_carries_nothing_secret():
    """It is unauthenticated on purpose — a check nobody can run without a
    token is a check nobody runs — so it must stay boring."""
    beat = Heartbeat(version="abc1234", started_at=time.time(), enabled=True)
    report = beat.report()
    flat = " ".join(f"{k}={v}" for k, v in report.items()).lower()
    for secret in ("key", "secret", "token", "srv", "/tmp", "password"):
        assert secret not in flat, report


# --- the endpoint ------------------------------------------------------------

def test_the_status_endpoint_answers_without_a_token():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.get("/x-bot/status")
    assert response.status_code == 200
    body = response.json()
    for field in ("version", "enabled", "healthy"):
        assert field in body, body


# --- the watcher's verdicts ---------------------------------------------------

@pytest.mark.parametrize("body, ok, says", [
    ({"version": "abc1234", "enabled": True, "healthy": True,
      "highlights": 16, "replies": 3, "seconds_since_poll": 8.0}, True, "ok"),
    ({"version": "abc1234", "enabled": False, "healthy": True}, True,
     "switched off"),
    ({"version": "abc1234", "enabled": True, "healthy": False,
      "highlights": 16, "seconds_since_poll": 400.0}, False, "NOT POLLING"),
    ({"version": "abc1234", "enabled": True, "healthy": True,
      "highlights": 0, "seconds_since_poll": 8.0}, False, "empty"),
    ({"version": "abc1234", "enabled": True, "healthy": True,
      "highlights": 16, "consecutive_errors": 5,
      "seconds_since_poll": 8.0}, False, "failed polls"),
    ({"enabled": False, "healthy": True, "version": "unknown",
      "note": "no heartbeat"}, False, "no heartbeat"),
])
def test_the_watcher_calls_each_state_correctly(body, ok, says):
    verdict, message = _run(_watcher(), body, expect=None)
    assert verdict is ok, message
    assert says in message, message


def test_the_watcher_catches_a_deploy_that_did_not_land():
    """The actual outage: healthy, polling, answering — on the old code."""
    watcher = _watcher()
    body = {"version": "old1111", "enabled": True, "healthy": True,
            "highlights": 16, "seconds_since_poll": 8.0}
    ok, message = _run(watcher, body, expect="new2222")
    assert not ok
    assert "did not land" in message


def test_an_unreachable_service_is_itself_the_alarm():
    watcher = _watcher()
    ok, message = watcher.check("http://127.0.0.1:9/nothing", None)
    assert not ok
    assert "unreachable" in message


def _run(watcher, body, expect):
    """Drive check() against a canned body without a network call."""
    import io
    import json
    import urllib.request

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    original = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: Response(
        json.dumps(body).encode())
    try:
        return watcher.check("http://example.invalid/x-bot/status", expect)
    finally:
        urllib.request.urlopen = original
