"""Turn a posted video clip into text, so the archive can place it.

A clipper posts thirty seconds of the show and captions it "must watch".
There is nothing to search on: no episode number, no quote, often no
words at all. The bot used to fall through to the newest episode and
answer with three thousand characters about it -- right only when the
clip happened to come from that show.

This reads the clip instead. Download the smallest rendition X offers,
take the audio, transcribe it, and hand the text to clipmatch.place,
which either names the episode or refuses.

Every failure here returns None rather than raising. This is called from
the reply loop, and a clip that cannot be read must mean "say nothing
about it" -- the bot then answers the way it did before, or stays quiet.
An exception would take down a poll cycle over a video somebody posted.

Groq rather than local whisper: the bot runs on Render, and mlx_whisper
needs Apple silicon and a laptop that happens to be awake. Measured on
this project, hosted turbo returns a 60-second clip in about a second.

Turbo rather than large-v3, which the archive ingest uses. Ingest is
quoting people in public and runs offline, so it buys the lower word
error rate with six times the wall clock. Placing a clip is a different
job: it needs enough correct eight-word runs to clear MIN_UNIQUE_RUNS,
typically finding two hundred where twelve are required, and it is doing
it while somebody waits for a reply.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

import httpx

from app.clipper import ffmpeg_available

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL = "whisper-large-v3-turbo"

# Only the opening minute is read. A clip long enough to need more is one
# whose first minute already carries far more than the twelve unique runs
# clipmatch asks for, and the cap is what stops somebody's four-hour
# upload becoming this bot's download.
MAX_CLIP_SECONDS = 60.0

# X hands back the smallest MP4 variant, which for a minute of video runs
# a few megabytes. The ceiling is a bound on a stranger's post rather than
# an expected size: without it, "the smallest variant" is still whatever
# the poster uploaded.
MAX_DOWNLOAD_BYTES = 48 * 1024 * 1024

# Groq's own ceiling is 25 MB. Sixty seconds of 16 kHz mono MP3 is about
# 250 KB, so this is three orders of magnitude of headroom and the check
# exists only to fail politely rather than post a doomed request.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024

_NETWORK_TIMEOUT = 30.0
_TRANSCRIBE_TIMEOUT = 90.0
_FFMPEG_TIMEOUT = 120


def usable(api_key: str | None) -> bool:
    """Whether reading a clip is possible at all on this deploy.

    Both halves are genuinely optional. A deploy without GROQ_API_KEY is
    the ordinary case on a fork, and ffmpeg is absent from plenty of
    environments including, until recently, this project's own CI.
    """
    return bool(api_key) and ffmpeg_available()


async def _download(url: str, dest: Path) -> bool:
    """The clip, up to the size ceiling. False when it cannot be had."""
    try:
        async with httpx.AsyncClient(timeout=_NETWORK_TIMEOUT,
                                     follow_redirects=True) as http:
            async with http.stream("GET", url) as response:
                response.raise_for_status()
                # Streamed and counted rather than read whole: content-length
                # is the poster's claim, and a response without one would
                # otherwise be unbounded.
                size = 0
                with open(dest, "wb") as fh:
                    async for chunk in response.aiter_bytes(65536):
                        size += len(chunk)
                        if size > MAX_DOWNLOAD_BYTES:
                            logger.info("clip is over %d MB; not reading it",
                                        MAX_DOWNLOAD_BYTES // (1024 * 1024))
                            return False
                        fh.write(chunk)
        return dest.exists() and dest.stat().st_size > 0
    except Exception as exc:                                # noqa: BLE001
        logger.info("could not download the clip: %s", exc)
        return False


def _to_audio(source: Path, dest: Path, seconds: float) -> bool:
    """The first `seconds` of audio as 16 kHz mono MP3.

    The same flags the archive ingest uses, so a clip is transcribed from
    exactly the shape of audio the indexed transcripts were made from.

    -ss before -i so ffmpeg seeks rather than decoding and discarding, and
    no memory cap: this decodes at most a minute of audio, nothing like
    the max-quality video encode that ulimit in clipper.py exists for.
    """
    try:
        done = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", "0", "-t", str(seconds),
             "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(dest)],
            capture_output=True, timeout=_FFMPEG_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info("ffmpeg could not read the clip: %s", exc)
        return False
    if done.returncode != 0:
        logger.info("ffmpeg refused the clip: %s",
                    (done.stderr or b"").decode(errors="replace")[-200:])
        return False
    return dest.exists() and dest.stat().st_size > 0


async def _transcribe(audio: Path, api_key: str) -> str | None:
    """The clip's words, or None.

    One attempt. transcribe_x_broadcast retries a 429 for up to eight
    rounds because it is indexing a five-hour show and the free tier's
    hourly allowance is genuinely smaller than the job. Here somebody is
    waiting for a reply, and waiting minutes to answer a clip is worse
    than not answering it -- the bot has other things to say.
    """
    if audio.stat().st_size > MAX_UPLOAD_BYTES:
        logger.info("clip audio is too large for the transcriber")
        return None
    try:
        with open(audio, "rb") as fh:
            async with httpx.AsyncClient(timeout=_TRANSCRIBE_TIMEOUT) as http:
                response = await http.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (audio.name, fh, "audio/mpeg")},
                    data={"model": MODEL, "response_format": "json",
                          "language": "en", "temperature": "0"},
                )
        if response.status_code == 429:
            logger.info("transcriber is rate limited; not reading this clip")
            return None
        response.raise_for_status()
        text = (response.json() or {}).get("text") or ""
    except Exception as exc:                                # noqa: BLE001
        logger.info("could not transcribe the clip: %s", exc)
        return None
    text = " ".join(text.split())
    return text or None


async def read(video_url: str, api_key: str | None, *,
               seconds: float = MAX_CLIP_SECONDS) -> str | None:
    """What is said in the first minute of a posted clip, or None.

    None on every failure, and None is the ordinary outcome: no key, no
    ffmpeg, a download that would not finish, a transcriber that is out
    of quota. The caller must treat it as "nothing to say about this
    clip" rather than as an error worth reporting to anybody.
    """
    if not usable(api_key) or not video_url:
        return None
    seconds = min(float(seconds or MAX_CLIP_SECONDS), MAX_CLIP_SECONDS)
    with tempfile.TemporaryDirectory(prefix="clipread-") as tmp:
        here = Path(tmp)
        video, audio = here / "clip.mp4", here / "clip.mp3"
        if not await _download(video_url, video):
            return None
        if not await asyncio.to_thread(_to_audio, video, audio, seconds):
            return None
        # to_thread because ffmpeg blocks, and this runs inside the web
        # process: a blocking decode here stops the site answering.
        return await _transcribe(audio, api_key)
