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

def test_a_missing_cookie_file_is_a_warning_not_a_failure(tmp_path, caplog):
    """A secret file that did not mount would otherwise fail every attempt
    with an error about the path, hiding the real problem."""
    import logging
    from app import clipper
    cmd_seen = {}

    def fake_run(cmd, **kw):
        cmd_seen["cmd"] = cmd
        raise RuntimeError("stop here")

    monkey = getattr(clipper, "subprocess")
    original = monkey.run
    monkey.run = fake_run
    try:
        with caplog.at_level(logging.WARNING):
            try:
                clipper.fetch_section("https://www.youtube.com/watch?v=x",
                                      0, 5, tmp_path / "out.mp4",
                                      cookies=str(tmp_path / "nope.txt"))
            except Exception:
                pass
    finally:
        monkey.run = original
    assert "--cookies" not in cmd_seen.get("cmd", [])
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
