from pathlib import Path
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


# --- audio that starts before the picture -----------------------------------

def _make(path, vstart=0.0, seconds=3, audio=True):
    """A tiny clip whose video optionally starts late, like a real section."""
    import subprocess
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i",
           f"testsrc=size=320x180:rate=30:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    if vstart:
        cmd += ["-itsoffset", str(vstart), "-map", "0:v"]
        if audio:
            cmd += ["-map", "1:a"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac"]
    cmd += [str(path)]
    subprocess.run(cmd, capture_output=True, timeout=120)


def test_a_section_with_no_audio_is_detected(tmp_path):
    """A section came back video-only and the render died on an audio
    bitrate flag with no stream to apply it to, which reads as a codec
    error rather than a missing stream."""
    from app.clipper import _has_audio
    silent = tmp_path / "silent.mp4"
    _make(silent, audio=False)
    withsound = tmp_path / "sound.mp4"
    _make(withsound, audio=True)
    assert _has_audio(silent) is False
    assert _has_audio(withsound) is True


def test_leading_gap_is_zero_when_the_streams_start_together(tmp_path):
    from app.clipper import leading_video_gap
    aligned = tmp_path / "aligned.mp4"
    _make(aligned)
    assert leading_video_gap(aligned) < 0.05


def test_leading_gap_never_goes_negative(tmp_path):
    """Video leading audio is something players handle; only the other
    direction needs correcting, so this must not report a trim."""
    from app.clipper import leading_video_gap
    f = tmp_path / "x.mp4"
    _make(f)
    assert leading_video_gap(f) >= 0.0


# --- a render that goes over budget must not take the service with it ------

def test_the_cap_wraps_the_command_without_mangling_arguments(monkeypatch):
    """A filename with a space in it must not become two arguments. That is
    the failure mode of building a shell string instead of exec "$@"."""
    import app.clipper as clipper
    monkeypatch.setattr(clipper.sys, "platform", "linux")
    cmd = ["ffmpeg", "-i", "/tmp/a file.mp4", "-vf", "scale=2:2", "/tmp/o.mp4"]
    wrapped = clipper._capped(cmd, 1400)
    assert wrapped[:2] == ["/bin/sh", "-c"]
    assert "ulimit -v 1433600" in wrapped[2]
    assert 'exec "$@"' in wrapped[2]
    # Everything after the sh placeholder is the original command, intact.
    assert wrapped[4:] == cmd


def test_the_cap_is_off_where_it_would_misfire(monkeypatch):
    """macOS counts mapped address space differently and a generous limit
    still refuses allocations ffmpeg makes routinely."""
    import app.clipper as clipper
    cmd = ["ffmpeg", "-i", "x.mp4", "y.mp4"]
    monkeypatch.setattr(clipper.sys, "platform", "darwin")
    assert clipper._capped(cmd, 1400) == cmd
    monkeypatch.setattr(clipper.sys, "platform", "linux")
    assert clipper._capped(cmd, 0) == cmd          # zero disables


@pytest.mark.parametrize("returncode,stderr,expected", [
    (137, "", True),                                # SIGKILL, the OOM killer
    (-9, "", True),
    (1, "Cannot allocate memory", True),
    (1, "std::bad_alloc", True),
    (1, "Error allocating a picture", True),
    (1, "Invalid argument", False),                 # a real filter error
    (1, "No such file or directory", False),
])
def test_out_of_memory_is_told_apart_from_a_broken_graph(returncode, stderr,
                                                         expected):
    """Reading an allocation failure as a filter bug is how an instance too
    small for the canvas gets mistaken for a code defect."""
    from app.clipper import _looks_like_out_of_memory
    assert _looks_like_out_of_memory(returncode, stderr) is expected


# --- what goes over the picture --------------------------------------------

from app.clipper import short_title                            # noqa: E402


@pytest.mark.parametrize("full,expected", [
    ("LIVE W/ WILL CLEMENTE, NET NET CAPITAL, & TYLER BERNABE: "
     "Market Bubble Ep 18 - Presented by @Polymarket", "Market Bubble Ep 18"),
    ("Why Ansem Thinks Ethereum Is Done.. | Market Bubble #4",
     "Market Bubble Ep 4"),
    ("Market Bubble: The Ansem Edition - Presented by @Polymarket",
     "The Ansem Edition"),
])
def test_the_title_on_the_clip_is_short(full, expected):
    """The full title ran across the broadcast's own Polymarket wordmark
    and neither could be read. A guest list is not what a viewer needs on
    a thirty-second clip; the caption carries that."""
    assert short_title(full) == expected


def test_an_unparseable_title_is_trimmed_not_dropped():
    long = "$100K POLYMARKET FANTASY FOOTBALL DRAFT NIGHT"
    out = short_title(long)
    assert out.endswith("…") and len(out) <= 31


def test_the_local_script_defaults_to_the_sites_canvas():
    """A clip cut locally and a clip a viewer cuts from the site have to be
    the same file. A silent bump to 1920 here meant a posted clip was 1080p
    while the button on the same moment gave 720p."""
    import re
    from app.clipper import CLIP_HEIGHT
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "make_clip.py").read_text()
    assert "default=CLIP_HEIGHT" in src
    assert not re.search(r'if args\.best and "--height" not in sys\.argv', src)


def test_memory_watching_falls_back_where_it_cannot_measure(monkeypatch):
    """Everywhere except Linux there is no VmPeak to read, and the render
    still has to run. Returning None is the honest answer; returning a
    number from a laptop is what broke production."""
    import app.clipper as clipper
    monkeypatch.setattr(clipper.sys, "platform", "darwin")
    done, peak = clipper.run_watching_memory(
        ["/bin/echo", "hello"], timeout=30)
    assert done.returncode == 0 and "hello" in done.stdout
    assert peak is None


def test_memory_watching_still_returns_output_and_status():
    """The measurement must not change what the caller sees. A render that
    fails has to report its stderr the same way it always did."""
    import app.clipper as clipper
    done, _ = clipper.run_watching_memory(
        ["/bin/sh", "-c", "echo out; echo err 1>&2; exit 3"], timeout=30)
    assert done.returncode == 3
    assert "out" in done.stdout and "err" in done.stderr


class TestClipsEndOnAFinishedSentence:
    """A clip that stops mid-word is the half a viewer is left holding.

    The first attempt at this only moved the cut to segment boundaries,
    which does nothing on a real transcript: Whisper ends a segment when
    it has heard enough audio, not when the speaker has finished, so the
    boundaries land mid-clause. On the episode this was built for, seven
    consecutive segments ended "a good trade right n-", "does have
    currentl-", "But one" -- not one of them a sentence. The clip still
    ended on "it's quantum resistant. It's also".

    So the stops are found inside the line and timed by how far through
    it they fall, which is the estimate build_captions already makes.
    """

    SEGS = [
        {"t": 0.0, "text": "so anyway that was the whole thing. and then"},
        {"t": 6.0, "text": "we got into it properly. i think the point is"},
        {"t": 12.0, "text": "that nobody checks. and honestly it's fine. but"},
        {"t": 18.0, "text": "there is one more thing worth saying here"},
    ]

    def test_it_does_not_stop_mid_clause(self):
        from app.clipper import snap_to_speech
        _, end = snap_to_speech(self.SEGS, 0.0, 16.0)
        # The line at 12s holds two stops -- "nobody checks." near its
        # start and "it's fine." near its end. Either is a real sentence;
        # what must not happen is stopping on the trailing "but".
        assert 13.0 < end < 18.0, end

    def test_a_boundary_is_only_a_fallback(self):
        # With no stop in reach the cut may still move to a boundary, but
        # it must never invent one that is not there.
        from app.clipper import snap_to_speech
        segs = [{"t": 0.0, "text": "no stops at all here"},
                {"t": 5.0, "text": "still nothing to end on"}]
        start, end = snap_to_speech(segs, 0.0, 40.0)
        assert (start, end) == (0.0, 40.0)

    def test_the_start_never_moves_forward(self):
        # Moving the start later would clip the moment being asked for.
        from app.clipper import snap_to_speech
        start, _ = snap_to_speech(self.SEGS, 7.5, 18.0)
        assert start <= 7.5

    def test_a_decimal_is_not_the_end_of_a_thought(self):
        from app.clipper import snap_to_speech
        segs = [{"t": 0.0, "text": "it went up 3.5 percent on the day and"},
                {"t": 6.0, "text": "kept going after that too"}]
        _, end = snap_to_speech(segs, 0.0, 5.0)
        # "3.5" is not a full stop, so the only candidate left is the
        # segment boundary. Landing before ~4s would mean it cut inside
        # the number.
        assert end >= 5.0, f"cut inside '3.5' at {end}"

    def test_empty_segments_change_nothing(self):
        from app.clipper import snap_to_speech
        assert snap_to_speech([], 3.0, 40.0) == (3.0, 40.0)
