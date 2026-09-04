import pytest


# --- pinning one exit IP ----------------------------------------------------

from app.clipper import pin_one_exit_ip                        # noqa: E402


# A rotating proxy gave yt-dlp one exit and ffmpeg another, so YouTube saw
# its own media URL used from an address it was not issued to and refused.
# Both processes read the same string, so the session belongs in it.
@pytest.mark.parametrize("proxy,session,expected", [
    # Nothing yet: the parameter block is opened with "__".
    ("http://user:pw@gw.dataimpulse.com:823", "abc123",
     "http://user__sessid.abc123:pw@gw.dataimpulse.com:823"),
    # Already carries a parameter: extended with ";", not replaced.
    ("http://user__cr.us:pw@gw.dataimpulse.com:823", "abc123",
     "http://user__cr.us;sessid.abc123:pw@gw.dataimpulse.com:823"),
    # Already names a session: left exactly alone.
    ("http://user__sessid.keep:pw@gw.x.com:823", "abc123",
     "http://user__sessid.keep:pw@gw.x.com:823"),
    # No credentials to hang it off, and nothing at all.
    ("http://noauth.example:8080", "abc123", "http://noauth.example:8080"),
    ("", "abc123", ""),
])
def test_pin_one_exit_ip(proxy, session, expected):
    assert pin_one_exit_ip(proxy, session) == expected


def test_pinning_never_drops_the_credentials():
    """A mangled proxy would fail as 407 and read as a flaky pool."""
    out = pin_one_exit_ip("http://user:pw@gw.dataimpulse.com:823", "s1")
    assert "pw@" in out and out.startswith("http://user__sessid.s1:")
