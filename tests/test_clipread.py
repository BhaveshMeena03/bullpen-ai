"""Reading a posted clip, and refusing to read one.

Almost every test here is a refusal, and that is the point. clipread is
called from the reply loop, where an exception takes down a poll cycle
over a video a stranger posted. Its contract is that every failure --
no key, no ffmpeg, a download that will not finish, a transcriber out
of quota -- returns None and the bot carries on.

The happy path needs a real Groq call, so it is stubbed. The refusals
are what actually protect the bot, and they are all reachable here.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app import clipread  # noqa: E402


@pytest.fixture(autouse=True)
def _ffmpeg_is_present(monkeypatch):
    """Pinned so the suite does not depend on the machine. The real
    ffmpeg_available is exercised by test_it_declines_without_ffmpeg."""
    monkeypatch.setattr(clipread, "ffmpeg_available", lambda: True)


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


class _Client:
    """Enough of httpx.AsyncClient for the two calls clipread makes."""

    def __init__(self, response, seen=None):
        self._response = response
        self._seen = seen

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, url, **kw):
        if self._seen is not None:
            self._seen["url"] = url
            self._seen.update(kw)
        return self._response


def _stub_post(monkeypatch, response, seen=None):
    monkeypatch.setattr(clipread.httpx, "AsyncClient",
                        lambda *a, **kw: _Client(response, seen))


# --- what makes it possible at all ------------------------------------------

class TestUsable:
    def test_no_key_means_no(self):
        assert clipread.usable(None) is False
        assert clipread.usable("") is False

    def test_it_declines_without_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(clipread, "ffmpeg_available", lambda: False)
        assert clipread.usable("a-key") is False

    def test_both_present_means_yes(self):
        assert clipread.usable("a-key") is True


# --- read(): the orchestration, and every way out of it ---------------------

class TestReadRefuses:
    def test_without_a_key(self):
        assert asyncio.run(clipread.read("https://x/v.mp4", None)) is None

    def test_without_a_url(self):
        assert asyncio.run(clipread.read("", "a-key")) is None

    def test_when_the_download_fails(self, monkeypatch):
        async def no(*_a, **_kw):
            return False
        monkeypatch.setattr(clipread, "_download", no)
        assert asyncio.run(clipread.read("https://x/v.mp4", "a-key")) is None

    def test_when_ffmpeg_fails(self, monkeypatch):
        async def yes(*_a, **_kw):
            return True
        monkeypatch.setattr(clipread, "_download", yes)
        monkeypatch.setattr(clipread, "_to_audio", lambda *a, **kw: False)
        assert asyncio.run(clipread.read("https://x/v.mp4", "a-key")) is None

    def test_when_the_transcriber_says_nothing(self, monkeypatch):
        async def yes(*_a, **_kw):
            return True

        async def nothing(*_a, **_kw):
            return None
        monkeypatch.setattr(clipread, "_download", yes)
        monkeypatch.setattr(clipread, "_to_audio", lambda *a, **kw: True)
        monkeypatch.setattr(clipread, "_transcribe", nothing)
        assert asyncio.run(clipread.read("https://x/v.mp4", "a-key")) is None


class TestReadSucceeds:
    def test_it_returns_what_was_said(self, monkeypatch):
        async def yes(*_a, **_kw):
            return True

        async def words(*_a, **_kw):
            return "the insiders benefited a lot from the launch"
        monkeypatch.setattr(clipread, "_download", yes)
        monkeypatch.setattr(clipread, "_to_audio", lambda *a, **kw: True)
        monkeypatch.setattr(clipread, "_transcribe", words)
        got = asyncio.run(clipread.read("https://x/v.mp4", "a-key"))
        assert got == "the insiders benefited a lot from the launch"

    def test_a_long_clip_is_capped_not_refused(self, monkeypatch):
        """A four-hour upload must cost one minute of transcription, not
        four hours of it. The cap is the only thing standing between this
        bot and whatever a stranger chose to post."""
        seen = {}

        async def yes(*_a, **_kw):
            return True

        async def words(*_a, **_kw):
            return "words"

        def to_audio(_src, _dest, seconds):
            seen["seconds"] = seconds
            return True
        monkeypatch.setattr(clipread, "_download", yes)
        monkeypatch.setattr(clipread, "_to_audio", to_audio)
        monkeypatch.setattr(clipread, "_transcribe", words)
        asyncio.run(clipread.read("https://x/v.mp4", "a-key", seconds=9999))
        assert seen["seconds"] == clipread.MAX_CLIP_SECONDS


# --- the download -----------------------------------------------------------

class _StreamResponse:
    def __init__(self, chunks, status_code=200):
        self.status_code = status_code
        self._chunks = chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")

    async def aiter_bytes(self, _size=65536):
        for chunk in self._chunks:
            yield chunk


class _StreamCtx:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *_exc):
        return False


class _StreamClient:
    def __init__(self, response, seen=None):
        self._response = response
        self._seen = seen

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def stream(self, method, url):
        if self._seen is not None:
            self._seen["method"], self._seen["url"] = method, url
        return _StreamCtx(self._response)


def _stub_stream(monkeypatch, response, seen=None):
    monkeypatch.setattr(clipread.httpx, "AsyncClient",
                        lambda *a, **kw: _StreamClient(response, seen))


class TestDownload:
    def test_it_writes_what_it_is_given(self, monkeypatch, tmp_path):
        _stub_stream(monkeypatch, _StreamResponse([b"abc", b"def"]))
        dest = tmp_path / "clip.mp4"
        assert asyncio.run(clipread._download("https://x/v.mp4", dest)) is True
        assert dest.read_bytes() == b"abcdef"

    def test_a_clip_over_the_ceiling_is_abandoned(self, monkeypatch, tmp_path):
        """The ceiling is a bound on a stranger's post, not on an expected
        size. Without it "the smallest variant X offers" is still whatever
        somebody chose to upload, and the bot downloads all of it."""
        monkeypatch.setattr(clipread, "MAX_DOWNLOAD_BYTES", 8)
        _stub_stream(monkeypatch, _StreamResponse([b"x" * 4] * 10))
        dest = tmp_path / "clip.mp4"
        assert asyncio.run(clipread._download("https://x/v.mp4", dest)) is False

    def test_it_stops_early_rather_than_reading_the_whole_thing(
            self, monkeypatch, tmp_path):
        """Counted as it streams, because content-length is the poster's
        claim and a response without one would otherwise be unbounded."""
        monkeypatch.setattr(clipread, "MAX_DOWNLOAD_BYTES", 8)
        pulled = []

        class _Counting(_StreamResponse):
            async def aiter_bytes(self, _size=65536):
                for i in range(1000):
                    pulled.append(i)
                    yield b"x" * 4

        _stub_stream(monkeypatch, _Counting([]))
        assert asyncio.run(
            clipread._download("https://x/v.mp4", tmp_path / "c.mp4")) is False
        assert len(pulled) < 10, f"kept pulling after the cap: {len(pulled)}"

    def test_a_404_is_a_refusal(self, monkeypatch, tmp_path):
        _stub_stream(monkeypatch, _StreamResponse([], status_code=404))
        assert asyncio.run(
            clipread._download("https://x/v.mp4", tmp_path / "c.mp4")) is False

    def test_a_network_error_is_a_refusal(self, monkeypatch, tmp_path):
        class _Boom:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            def stream(self, *_a, **_kw):
                raise OSError("connection reset")
        monkeypatch.setattr(clipread.httpx, "AsyncClient",
                            lambda *a, **kw: _Boom())
        assert asyncio.run(
            clipread._download("https://x/v.mp4", tmp_path / "c.mp4")) is False

    def test_an_empty_body_is_a_refusal(self, monkeypatch, tmp_path):
        """A zero-byte file is not a clip, and handing it to ffmpeg gets a
        confusing codec error instead of "there was nothing there"."""
        _stub_stream(monkeypatch, _StreamResponse([]))
        assert asyncio.run(
            clipread._download("https://x/v.mp4", tmp_path / "c.mp4")) is False

    def test_it_follows_redirects(self, monkeypatch, tmp_path):
        """X's media URLs redirect, and a client that does not follow them
        downloads the redirect body and calls it a video."""
        seen = {}
        monkeypatch.setattr(
            clipread.httpx, "AsyncClient",
            lambda *a, **kw: seen.update(kw) or _StreamClient(
                _StreamResponse([b"data"])))
        asyncio.run(clipread._download("https://x/v.mp4", tmp_path / "c.mp4"))
        assert seen.get("follow_redirects") is True


# --- the audio step ---------------------------------------------------------

class TestToAudio:
    def _cmd(self, monkeypatch, returncode=0, write=True):
        seen = {}

        def fake_run(cmd, **_kw):
            seen["cmd"] = cmd
            if write:
                Path(cmd[-1]).write_bytes(b"audio")

            class _Done:
                pass
            done = _Done()
            done.returncode = returncode
            done.stderr = b"ffmpeg said no"
            return done
        monkeypatch.setattr(clipread.subprocess, "run", fake_run)
        return seen

    def test_it_asks_for_the_shape_the_archive_was_built_from(
            self, monkeypatch, tmp_path):
        """16 kHz mono, the same flags the ingest uses. A clip transcribed
        from differently-shaped audio is a clip matched against runs made
        from something else."""
        seen = self._cmd(monkeypatch)
        assert clipread._to_audio(tmp_path / "in.mp4",
                                  tmp_path / "out.mp3", 60.0) is True
        cmd = seen["cmd"]
        assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
        assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"
        assert "-vn" in cmd, "the video track is never used"
        assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "60.0"

    def test_a_refusing_ffmpeg_is_not_an_exception(self, monkeypatch, tmp_path):
        self._cmd(monkeypatch, returncode=1)
        assert clipread._to_audio(tmp_path / "in.mp4",
                                  tmp_path / "out.mp3", 60.0) is False

    def test_a_silent_failure_that_writes_nothing_is_caught(
            self, monkeypatch, tmp_path):
        """returncode 0 and no file is the shape of a silent no-op, and
        the caller would otherwise hand a missing path to the uploader."""
        self._cmd(monkeypatch, returncode=0, write=False)
        assert clipread._to_audio(tmp_path / "in.mp4",
                                  tmp_path / "out.mp3", 60.0) is False

    def test_a_missing_binary_is_not_an_exception(self, monkeypatch, tmp_path):
        def boom(*_a, **_kw):
            raise FileNotFoundError("ffmpeg")
        monkeypatch.setattr(clipread.subprocess, "run", boom)
        assert clipread._to_audio(tmp_path / "in.mp4",
                                  tmp_path / "out.mp3", 60.0) is False

    def test_a_hanging_ffmpeg_is_not_an_exception(self, monkeypatch, tmp_path):
        def slow(*_a, **_kw):
            raise subprocess.TimeoutExpired("ffmpeg", 120)
        monkeypatch.setattr(clipread.subprocess, "run", slow)
        assert clipread._to_audio(tmp_path / "in.mp4",
                                  tmp_path / "out.mp3", 60.0) is False


# --- the transcriber --------------------------------------------------------

class TestTranscribe:
    def _audio(self, tmp_path, size=2048):
        f = tmp_path / "clip.mp3"
        f.write_bytes(b"x" * size)
        return f

    def test_it_returns_the_text(self, monkeypatch, tmp_path):
        _stub_post(monkeypatch, _Response(200, {"text": "  hello   there "}))
        got = asyncio.run(clipread._transcribe(self._audio(tmp_path), "k"))
        assert got == "hello there", "whitespace is collapsed"

    def test_rate_limited_is_a_refusal_not_a_wait(self, monkeypatch, tmp_path):
        """transcribe_x_broadcast retries a 429 for eight rounds because it
        is indexing a five-hour show offline. Here somebody is waiting for
        a reply, and the bot has other things to say."""
        _stub_post(monkeypatch, _Response(429))
        assert asyncio.run(
            clipread._transcribe(self._audio(tmp_path), "k")) is None

    def test_a_server_error_is_a_refusal(self, monkeypatch, tmp_path):
        _stub_post(monkeypatch, _Response(500))
        assert asyncio.run(
            clipread._transcribe(self._audio(tmp_path), "k")) is None

    def test_empty_text_is_none_not_an_empty_string(self, monkeypatch, tmp_path):
        """clipmatch.place("") would score nothing and refuse anyway, but
        an empty string reads as "it was read and said nothing" rather
        than "it was not read"."""
        _stub_post(monkeypatch, _Response(200, {"text": "   "}))
        assert asyncio.run(
            clipread._transcribe(self._audio(tmp_path), "k")) is None

    def test_a_network_error_is_a_refusal(self, monkeypatch, tmp_path):
        class _Boom:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def post(self, *_a, **_kw):
                raise OSError("connection reset")
        monkeypatch.setattr(clipread.httpx, "AsyncClient",
                            lambda *a, **kw: _Boom())
        assert asyncio.run(
            clipread._transcribe(self._audio(tmp_path), "k")) is None

    def test_audio_over_the_ceiling_is_never_uploaded(self, monkeypatch,
                                                      tmp_path):
        """Refused before the request, so a doomed upload is not attempted
        against somebody's quota."""
        posted = {"called": False}

        class _NeverCalled(_Client):
            async def post(self, *_a, **_kw):
                posted["called"] = True
                return _Response(200, {"text": "should not happen"})
        monkeypatch.setattr(clipread.httpx, "AsyncClient",
                            lambda *a, **kw: _NeverCalled(_Response()))
        big = self._audio(tmp_path, size=clipread.MAX_UPLOAD_BYTES + 1)
        assert asyncio.run(clipread._transcribe(big, "k")) is None
        assert posted["called"] is False

    def test_it_sends_the_key_and_the_model(self, monkeypatch, tmp_path):
        seen = {}
        _stub_post(monkeypatch, _Response(200, {"text": "hi"}), seen)
        asyncio.run(clipread._transcribe(self._audio(tmp_path), "secret-key"))
        assert seen["url"] == clipread.GROQ_URL
        assert seen["headers"]["Authorization"] == "Bearer secret-key"
        assert seen["data"]["model"] == clipread.MODEL
        assert seen["data"]["temperature"] == "0"
