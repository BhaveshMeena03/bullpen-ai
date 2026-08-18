"""Turn a moment into a downloadable video clip, on the server.

The page can already point at a moment and hand over a still card. What
people actually post is a short video with the words burned on, and asking
them to run a Python script is the same as not offering it.

Everything expensive about a clipper is already paid for here. The
transcript was indexed with timestamps months ago, so captions come out of
the index rather than out of a speech-to-text pass, and only the seconds
being clipped are ever downloaded — a minute of a three-hour episode is a
couple of megabytes, not the whole file.

What it costs, measured rather than guessed: a 55-second clip at 720x720
encodes in 3.8s of single-core CPU and comes out at 9MB. CPU is therefore
not the binding constraint; egress is. 9MB a clip against 5GB of included
bandwidth is roughly 550 clips a month, so the limits below exist to make
a viral afternoon cost a refusal rather than a bill.

Deliberately modest defaults:

  720x720   half the bytes of 1080 and indistinguishable on a phone
  45s max   long enough for a real exchange, short enough to stay cheap
  1 at a time  video encoding is the most expensive thing this service can
               be asked to do; concurrency here is how a small box dies

Jobs run in the background and the client polls, because even a fast
encode is far longer than a request should be held open.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SIZE = 720
BG = "#0b0e11"
GREEN = "#16c784"
CAPTION_WORDS = 5

MAX_CLIP_SECONDS = 45
MIN_CLIP_SECONDS = 5
# Only one encode at a time. The queue is what keeps a burst from turning
# into an out-of-memory kill on a small instance.
MAX_CONCURRENT = 1
# Finished clips are deleted after this. The disk is ephemeral anyway, and
# a clip nobody fetched in half an hour is one nobody wanted.
CLIP_TTL_SECONDS = 1800
MAX_JOBS_TRACKED = 200

# yt-dlp negotiates the media URL as one client and hands it to ffmpeg,
# which fetches it as itself; most clients tie the URL to the caller, so
# ffmpeg gets a 403 and the download fails as "ffmpeg exited with code 8".
# web_safari, web and tv refuse the format outright. mweb works.
PLAYER_CLIENT = "mweb"

# Fonts have to be found on both a Mac and a slim Debian image; neither
# has the other's. Missing fonts degrade to Pillow's bitmap default, which
# looks broken rather than plain, so the image installs DejaVu explicitly.
FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]
FONT_CANDIDATES_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]


def _font(candidates: list[str], size: int):
    from PIL import ImageFont
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    logger.warning("No TrueType font found; captions will look wrong.")
    return ImageFont.load_default()


def _encoder() -> list[str]:
    """Hardware encode on a Mac, x264 on the server.

    Checked once against the actual binary rather than assumed from the
    platform: the ffmpeg in the container is a different build from the
    one on a laptop, and guessing wrong fails at render time.
    """
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        out = ""
    if "h264_videotoolbox" in out:
        return ["-c:v", "h264_videotoolbox", "-b:v", "2500k"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-b:v", "1200k",
            "-pix_fmt", "yuv420p"]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# ─── captions ─────────────────────────────────────────────────────────────

def build_captions(segments: list[dict], start: float,
                   end: float) -> list[tuple[float, float, str]]:
    """Caption lines for the window, re-timed so 0 is the clip's start.

    The transcript arrives in ~8 second blocks, far too much text to read
    at once on a phone. Each block is split into short groups and the
    block's duration shared out by character count. An approximation, but
    the captions are auto-generated to begin with: legible and roughly in
    sync is the honest goal, not frame accuracy.
    """
    out: list[tuple[float, float, str]] = []
    window = [s for s in segments if start - 12 <= s.get("t", 0) < end]
    for i, seg in enumerate(window):
        seg_start = float(seg["t"])
        seg_end = float(window[i + 1]["t"]) if i + 1 < len(window) else seg_start + 8
        words = str(seg.get("text", "")).split()
        if not words:
            continue
        chunks = [" ".join(words[j:j + CAPTION_WORDS])
                  for j in range(0, len(words), CAPTION_WORDS)]
        total = max(sum(len(c) for c in chunks), 1)
        cursor = seg_start
        for chunk in chunks:
            share = (seg_end - seg_start) * (len(chunk) / total)
            a, b = cursor - start, cursor + share - start
            cursor += share
            if b <= 0 or a >= end - start:
                continue
            out.append((max(a, 0.0), min(b, end - start), chunk))
    return out


# ─── drawing ──────────────────────────────────────────────────────────────
#
# All text is drawn with Pillow and composited through `overlay`, rather
# than with ffmpeg's drawtext and subtitles filters. Those need libfreetype
# and libass, and the ffmpeg used while building this had neither, so they
# simply did not exist as filters. Depending on how someone compiled their
# ffmpeg is a bad foundation; overlay is in every build.

def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) > max_width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def _outlined(draw, xy, text, font, fill, outline=3):
    """Outline, not a caption box: a box is easier to read but covers the
    faces, and the faces are why a clip beats a screenshot."""
    x, y = xy
    for dx in range(-outline, outline + 1):
        for dy in range(-outline, outline + 1):
            if dx * dx + dy * dy <= outline * outline:
                draw.text((x + dx, y + dy), text, font=font, fill="black")
    draw.text((x, y), text, font=font, fill=fill)


def make_backdrop(title: str, stamp: str, path: Path) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (SIZE, SIZE), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, SIZE, 7], fill=GREEN)

    font = _font(FONT_CANDIDATES_BOLD, 27)
    y = 34
    for line in _wrap(draw, title, font, SIZE - 90)[:2]:
        w = draw.textlength(line, font=font)
        draw.text(((SIZE - w) / 2, y), line, font=font, fill="#e6e8ea")
        y += 34

    foot_font = _font(FONT_CANDIDATES_REGULAR, 18)
    foot = f"{stamp}   ·   search.lexthedev.com"
    w = draw.textlength(foot, font=foot_font)
    draw.text(((SIZE - w) / 2, SIZE - 44), foot, font=foot_font, fill=GREEN)
    img.save(path)


def make_caption(text: str, path: Path) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _font(FONT_CANDIDATES_BOLD, 35)
    lines = _wrap(draw, text, font, SIZE - 100)[:3]
    y = SIZE - 168 - (len(lines) - 1) * 40
    for line in lines:
        w = draw.textlength(line, font=font)
        _outlined(draw, ((SIZE - w) / 2, y), line, font, "white")
        y += 40
    img.save(path)


# ─── the work ─────────────────────────────────────────────────────────────

def fetch_section(url: str, start: float, end: float, dest: Path,
                  proxy: str | None = None) -> None:
    cmd = ["yt-dlp", "--quiet", "--no-warnings",
           "--download-sections", f"*{start:.2f}-{end:.2f}",
           "--force-keyframes-at-cuts",
           "-f", "bv*[height<=720]+ba/b[height<=720]/b",
           "--extractor-args", f"youtube:player_client={PLAYER_CLIENT}",
           "--remote-components", "ejs:github"]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd += ["-o", str(dest), url]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not dest.exists():
        tail = ((result.stderr or "").strip().splitlines() or ["(no stderr)"])[-1]
        raise RuntimeError(f"could not fetch that section: {tail[:180]}")


def render(source: Path, captions, backdrop: Path,
           workdir: Path, out: Path) -> None:
    inputs = ["-i", str(source), "-loop", "1", "-i", str(backdrop)]
    steps = [f"[0:v]scale={SIZE}:-2[vid]",
             "[1:v][vid]overlay=(W-w)/2:(H-h)/2:shortest=1[base]"]
    label = "base"
    for i, (start, end, text) in enumerate(captions):
        png = workdir / f"cap{i:04d}.png"
        make_caption(text, png)
        inputs += ["-i", str(png)]
        nxt = f"c{i}"
        steps.append(f"[{label}][{i + 2}:v]overlay=0:0:"
                     f"enable='between(t,{start:.2f},{end:.2f})'[{nxt}]")
        label = nxt

    result = subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(steps),
         "-map", f"[{label}]", "-map", "0:a?", *_encoder(),
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
         "-shortest", str(out)],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        tail = "; ".join((result.stderr or "").strip().splitlines()[-3:])
        raise RuntimeError(f"render failed: {tail[:220]}")


def stamp(seconds: float) -> str:
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ─── jobs ─────────────────────────────────────────────────────────────────

@dataclass
class Job:
    id: str
    status: str = "queued"          # queued | working | done | failed
    error: str | None = None
    path: Path | None = None
    created: float = field(default_factory=time.monotonic)
    title: str = ""
    seconds: float = 0.0


class ClipService:
    """Background clip jobs, one at a time, with everything bounded."""

    def __init__(self, proxy: str | None = None):
        self._jobs: dict[str, Job] = {}
        self._gate = asyncio.Semaphore(MAX_CONCURRENT)
        self._proxy = proxy
        self._dir = Path(tempfile.gettempdir()) / "clips"
        self._dir.mkdir(exist_ok=True)

    # -- lifecycle ---------------------------------------------------------

    def _sweep(self) -> None:
        now = time.monotonic()
        for job_id, job in list(self._jobs.items()):
            if now - job.created > CLIP_TTL_SECONDS:
                if job.path and job.path.exists():
                    job.path.unlink(missing_ok=True)
                self._jobs.pop(job_id, None)
        while len(self._jobs) > MAX_JOBS_TRACKED:
            oldest = min(self._jobs.values(), key=lambda j: j.created)
            if oldest.path:
                oldest.path.unlink(missing_ok=True)
            self._jobs.pop(oldest.id, None)

    def get(self, job_id: str) -> Job | None:
        self._sweep()
        return self._jobs.get(job_id)

    def queued_count(self) -> int:
        return sum(1 for j in self._jobs.values()
                   if j.status in ("queued", "working"))

    # -- the job -----------------------------------------------------------

    def submit(self, episode: dict, start: float, end: float) -> Job:
        self._sweep()
        job = Job(id=uuid.uuid4().hex[:12], title=episode.get("title", ""),
                  seconds=end - start)
        self._jobs[job.id] = job
        asyncio.create_task(self._run(job, episode, start, end))
        return job

    async def _run(self, job: Job, episode: dict,
                   start: float, end: float) -> None:
        async with self._gate:
            job.status = "working"
            try:
                out = self._dir / f"{job.id}.mp4"
                # Blocking work: yt-dlp and ffmpeg are subprocesses, and
                # running them inline would stall the event loop for every
                # other request on this single-worker server.
                await asyncio.to_thread(self._build, episode, start, end, out)
                job.path = out
                job.status = "done"
                logger.info("clip %s done (%.0fs, %.1fMB)", job.id,
                            end - start, out.stat().st_size / 1e6)
            except Exception as exc:  # noqa: BLE001
                job.status = "failed"
                job.error = str(exc)[:200]
                logger.warning("clip %s failed: %s", job.id, job.error)

    def _build(self, episode: dict, start: float,
               end: float, out: Path) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            raw = work / "raw.mp4"
            fetch_section(episode["url"], start, end, raw, self._proxy)
            backdrop = work / "backdrop.png"
            make_backdrop(episode.get("title", ""), stamp(start), backdrop)
            captions = build_captions(episode.get("segments", []), start, end)
            render(raw, captions, backdrop, work, out)
