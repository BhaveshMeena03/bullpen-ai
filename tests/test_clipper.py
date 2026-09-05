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


# --- cookies, when a signed-in session is the only thing that works --------

def test_a_read_only_secret_is_copied_somewhere_writable(tmp_path):
    """yt-dlp rewrites the jar after every request, because YouTube rotates
    the session. A secret file is mounted read-only, so pointing straight
    at it failed every attempt with "[Errno 30] Read-only file system" --
    correct cookies that never got used."""
    from pathlib import Path
    from app.clipper import _writable_cookie_jar
    secret = tmp_path / "cookies.txt"
    secret.write_text("# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tx\n")
    secret.chmod(0o444)
    jar = _writable_cookie_jar(str(secret))
    assert jar is not None and jar != str(secret)
    copy = Path(jar)
    assert copy.read_text() == secret.read_text()
    copy.write_text("refreshed")          # the point: writable
    assert copy.read_text() == "refreshed"


def test_no_cookie_setting_means_no_cookies():
    from app.clipper import _writable_cookie_jar
    assert _writable_cookie_jar(None) is None
    assert _writable_cookie_jar("") is None


def test_a_missing_cookie_file_is_a_warning_not_a_failure(tmp_path, caplog):
    """A secret file that did not mount would otherwise fail every attempt
    with an error about the path, hiding the real problem."""
    import logging
    from app.clipper import _writable_cookie_jar
    with caplog.at_level(logging.WARNING):
        assert _writable_cookie_jar(str(tmp_path / "nope.txt")) is None
    assert any("no file there" in r.getMessage() for r in caplog.records)


def test_a_cookie_file_that_exists_is_passed_to_yt_dlp(tmp_path):
    from app import clipper
    jar = tmp_path / "cookies.txt"
    jar.write_text("# Netscape HTTP Cookie File\n")
    cmd_seen = {}

    def fake_run(cmd, **kw):
        cmd_seen["cmd"] = cmd
        raise RuntimeError("stop here")

    original = clipper.subprocess.run
    clipper.subprocess.run = fake_run
    try:
        try:
            clipper.fetch_section("https://www.youtube.com/watch?v=x",
                                  0, 5, tmp_path / "out.mp4", cookies=str(jar))
        except Exception:
            pass
    finally:
        clipper.subprocess.run = original
    cmd = cmd_seen.get("cmd", [])
    assert "--cookies" in cmd and str(jar) in cmd
