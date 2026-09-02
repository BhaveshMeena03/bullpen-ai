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

# The canvas is a parameter, not a constant. A clip made on a laptop should
# be the best the source allows; one made on a small server has to be the
# cheapest thing that still looks right on a phone. Same code, one number.
DEFAULT_SIZE = 720          # server default: half the bytes of 1080
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
# web_safari, web and tv refuse the format outright. mweb works — and that
# is the whole reason it was chosen, which turned out to be the bug.
#
# mweb is served ONE progressive format: 640x360. Not a preference among
# several, the only thing on offer. So every clip ever cut from a YouTube
# episode was sourced at 360p and then scaled UP to a 720 or 1080 canvas,
# and asking for a bigger canvas made it blurrier rather than sharper.
# Nothing in the pipeline could have recovered it; the detail was never
# downloaded.
#
# Empty means yt-dlp picks, and what it picks offers the real ladder up to
# 1920x1080 60fps. The 403 this was working around comes from handing a
# client-bound URL to ffmpeg, which only happens on the streaming path —
# the download here writes a file first, so it does not apply.
PLAYER_CLIENT = ""

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


def _encoder(size: int = DEFAULT_SIZE, best: bool = False) -> list[str]:
    """Hardware encode on a Mac, x264 on the server.

    Checked once against the actual binary rather than assumed from the
    platform: the ffmpeg in the container is a different build from the
    one on a laptop, and guessing wrong fails at render time.

    Bitrate scales with the canvas. 2500k is right for 720 and visibly
    soft at 1080, and the only reason to hold 1080 down would be an egress
    bill that a local run does not have.

    `best` drops the hardware path entirely and encodes for quality rather
    than for a number of bits per second. VideoToolbox is fast and its
    fixed-bitrate mode is the weaker of the two ways to encode this: a
    static backdrop with a small moving panel spends its budget evenly
    whether the frame needs it or not, so the faces get what is left. x264
    at a constant quality spends bits where the picture is, and the render
    is a few seconds slower for something that gets re-encoded once by X
    and then watched. Only worth it for a clip that goes out publicly.
    """
    if best:
        return ["-c:v", "libx264", "-preset", "slow", "-crf", "18",
                "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2"]
    rate = f"{int(2500 * (size / DEFAULT_SIZE) ** 2)}k"
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        out = ""
    if "h264_videotoolbox" in out:
        return ["-c:v", "h264_videotoolbox", "-b:v", rate]
    return ["-c:v", "libx264", "-preset", "veryfast", "-b:v", rate,
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


def make_backdrop(title: str, stamp: str, path: Path,
                  size: int = DEFAULT_SIZE) -> None:
    from PIL import Image, ImageDraw

    k = size / DEFAULT_SIZE          # type scales with the canvas
    img = Image.new("RGB", (size, size), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size, int(7 * k)], fill=GREEN)

    font = _font(FONT_CANDIDATES_BOLD, int(27 * k))
    y = 34 * k
    for line in _wrap(draw, title, font, size - 90 * k)[:2]:
        w = draw.textlength(line, font=font)
        draw.text(((size - w) / 2, y), line, font=font, fill="#e6e8ea")
        y += 34 * k

    foot_font = _font(FONT_CANDIDATES_REGULAR, int(18 * k))
    foot = f"{stamp}   ·   search.lexthedev.com"
    w = draw.textlength(foot, font=foot_font)
    draw.text(((size - w) / 2, size - 44 * k), foot, font=foot_font, fill=GREEN)
    img.save(path)


def make_wide_overlay(title: str, stamp: str, path: Path,
                      width: int = 1920, height: int = 1080) -> None:
    """Title and credit painted ON the picture, for a 16:9 clip.

    The square layout parks the video in a band and fills the space above
    and below it with a title and a credit. That reads well in a feed, and
    it means anything playing the file in a 16:9 window pillarboxes it —
    black down both sides, which is what people actually notice.

    Here the picture fills the frame and the text sits on top of it. The
    only furniture is a scrim behind each line, dark enough to read against
    a bright frame and short enough to leave the faces alone.
    """
    from PIL import Image, ImageDraw

    k = width / 1920                 # 1.0 at 1080p, and scales from there
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, int(7 * k)], fill=GREEN)

    # A gradient would be nicer than a flat band and costs a loop over every
    # row; behind text this size the difference is not visible.
    # Opaque enough to actually cover. At 165 the broadcast's own
    # "LOS ANGELES 1:42 PM PT" read straight through the credit.
    scrim = Image.new("RGBA", (width, int(74 * k)), (0, 0, 0, 225))
    img.paste(scrim, (0, int(7 * k)), scrim)

    # Shrink to fit rather than crop. Taking the first wrapped line cut
    # "Why Ansem Thinks Ethereum Is Done.. | Market" and threw away the
    # episode number, which is the half a reader needs.
    foot_font = _font(FONT_CANDIDATES_REGULAR, int(24 * k))
    room = width - 68 * k - draw.textlength(
        f"{stamp}  ·  search.lexthedev.com", font=foot_font) - 40 * k
    for pt in (36, 33, 30, 27, 24):
        font = _font(FONT_CANDIDATES_BOLD, int(pt * k))
        if draw.textlength(title, font=font) <= room:
            break
    shown = title
    while draw.textlength(shown, font=font) > room and len(shown) > 12:
        shown = shown[:-2]
    if shown != title:
        shown = shown.rstrip(" .|-") + "…"
    draw.text((34 * k, (74 * k - pt * k) / 2 + 7 * k), shown,
              font=font, fill="#e6e8ea")

    # In the top band beside the title, not along the bottom. The
    # broadcast runs its own logo, chyron and ticker across the lower third
    # of every frame, so anything put down there is competing with three
    # things at once — and the captions have to live there too.
    foot = f"{stamp}  ·  search.lexthedev.com"
    fw = draw.textlength(foot, font=foot_font)
    draw.text((width - fw - 34 * k, (74 * k - 24 * k) / 2 + 7 * k),
              foot, font=foot_font, fill=GREEN)

    img.save(path)


def make_caption(text: str, path: Path, size: int = DEFAULT_SIZE,
                 height: int | None = None) -> None:
    from PIL import Image, ImageDraw

    k = size / DEFAULT_SIZE
    height = height or size
    # Type is sized against the SHORT edge, not the width. Scaling it off
    # the width put 93px lettering on a 1080-tall frame — nearly a tenth of
    # the picture, when a subtitle wants about a twentieth. On the square,
    # where the two are equal, nothing changes.
    kt = min(size, height) / DEFAULT_SIZE
    img = Image.new("RGBA", (size, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _font(FONT_CANDIDATES_BOLD, int(35 * kt))
    lines = _wrap(draw, text, font, size - 100 * k)[:3]
    # Measured up from the bottom of whatever frame this is. On the square
    # the gap under the picture is where the caption lives, so it is deep.
    # On a 16:9 frame the picture goes to the edge and the same gap would
    # park the words across the faces. They go to the very bottom instead,
    # over the broadcast's own ticker, which is the one strip of that
    # frame nothing is lost by covering.
    pad = (168 * k) if height >= size else (0.10 * height)
    step = 44 * kt
    y = height - pad - (len(lines) - 1) * step

    # A soft panel behind the words, only on the wide frame. There the
    # captions sit on the picture, and the broadcast's ticker underneath
    # them is bright, busy and moving — an outline alone has to fight it
    # every frame. On the square they sit on flat backdrop and need
    # nothing, which is what the outline was written for.
    if height < size and lines:
        widest = max(draw.textlength(ln, font=font) for ln in lines)
        bx, by = 34 * kt, 16 * kt
        panel = Image.new(
            "RGBA",
            (int(widest + bx * 2), int(len(lines) * step + by * 2)),
            (0, 0, 0, 110))
        img.paste(panel,
                  (int((size - widest) / 2 - bx), int(y - by)), panel)

    for line in lines:
        w = draw.textlength(line, font=font)
        _outlined(draw, ((size - w) / 2, y), line, font, "white",
                  outline=max(2, int(2.5 * kt)))
        y += step
    img.save(path)


# ─── the work ─────────────────────────────────────────────────────────────

def _ytdlp_binary() -> str:
    """Prefer the venv's yt-dlp over whatever is on PATH.

    In the container they are the same thing. Run from a script under a
    different interpreter they are not, and a bare "yt-dlp" raises
    FileNotFoundError — which is how moving this function out of the
    script and into the shared module broke the script.
    """
    local = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "yt-dlp"
    if local.exists():
        return str(local)
    return shutil.which("yt-dlp") or "yt-dlp"


def fetch_section(url: str, start: float, end: float, dest: Path,
                  proxy: str | None = None, height: int = DEFAULT_SIZE) -> None:
    """Download just the requested seconds of a video, at most `height` tall.

    Works for YouTube and for an X broadcast replay. The two need different
    handling in two places:

      player client   YouTube refuses a section download from some clients;
                      mweb is the one that works. X has no such argument and
                      rejects the whole flag, so it is only passed to
                      YouTube URLs.
      fragments       An X replay arrives as thousands of small HLS
                      fragments fetched one at a time by default, which is
                      bounded by round-trip latency rather than bandwidth.
                      Sixteen at once is the difference between seconds and
                      minutes even for a short section.
    """
    is_youtube = "youtube.com" in url or "youtu.be" in url
    cmd = [_ytdlp_binary(), "--quiet", "--no-warnings",
           "--download-sections", f"*{start:.2f}-{end:.2f}",
           "--force-keyframes-at-cuts",
           # H.264 first so the merge stays an mp4. Left to itself yt-dlp
           # takes AV1 with Opus, which is a smaller download and a webm,
           # and then everything downstream is decoding AV1 for no benefit
           # — this gets re-encoded on the next line anyway.
           "-f", (f"bv*[height<={height}][vcodec^=avc1]+ba[ext=m4a]/"
                  f"bv*[height<={height}]+ba/b[height<={height}]/b"),
           "--concurrent-fragments", "16"]
    if is_youtube:
        if PLAYER_CLIENT:
            cmd += ["--extractor-args",
                    f"youtube:player_client={PLAYER_CLIENT}"]
        cmd += ["--remote-components", "ejs:github"]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd += ["-o", str(dest), url]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not dest.exists():
        tail = ((result.stderr or "").strip().splitlines() or ["(no stderr)"])[-1]
        raise RuntimeError(f"could not fetch that section: {tail[:180]}")


def render(source: Path, captions, backdrop: Path, workdir: Path,
           out: Path, size: int = DEFAULT_SIZE, best: bool = False,
           wide: bool = False) -> None:
    """Compose the clip.

    Square by default: the picture sits in a band with the title above it
    and the credit below. `wide` fills a 16:9 frame with the picture and
    lays the same text over it, which is the layout to use when the clip
    will be watched rather than scrolled past — a square file pillarboxes
    in any 16:9 player, and the black down both sides is the first thing
    anyone notices.
    """
    height = int(size * 9 / 16) if wide else size
    inputs = ["-i", str(source), "-loop", "1", "-i", str(backdrop)]
    if wide:
        # Cover, not fit: fill the frame and crop the overflow rather than
        # leaving a bar. The source is already 16:9, so this crops nothing
        # in practice and protects the frame if one ever is not.
        steps = [f"[0:v]scale={size}:{height}:force_original_aspect_ratio="
                 f"increase,crop={size}:{height}[vid]",
                 "[vid][1:v]overlay=0:0:shortest=1[base]"]
    else:
        steps = [f"[0:v]scale={size}:-2[vid]",
                 "[1:v][vid]overlay=(W-w)/2:(H-h)/2:shortest=1[base]"]
    label = "base"
    for i, (start, end, text) in enumerate(captions):
        png = workdir / f"cap{i:04d}.png"
        make_caption(text, png, size, height)
        inputs += ["-i", str(png)]
        nxt = f"c{i}"
        steps.append(f"[{label}][{i + 2}:v]overlay=0:0:"
                     f"enable='between(t,{start:.2f},{end:.2f})'[{nxt}]")
        label = nxt

    result = subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(steps),
         "-map", f"[{label}]", "-map", "0:a?", *_encoder(size, best),
         "-c:a", "aac", "-b:a", "192k" if best else "128k",
         "-movflags", "+faststart",
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
