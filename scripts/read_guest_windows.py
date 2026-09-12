"""Who was on air, and when -- read off the broadcast itself.

    .venv/bin/python scripts/read_guest_windows.py --only 2uOI23N1yXY
    .venv/bin/python scripts/read_guest_windows.py --only 2uOI23N1yXY --keep
    .venv/bin/python scripts/read_guest_windows.py

The transcripts say what was said and the voiceprints say which voice
said it, and between them they still cannot name a guest. Voices are
matched by recurrence -- the two people in every show are the hosts --
and a guest who appears once has nothing to recur against. That is why
92,979 segments carry seven names.

The show has been captioning itself the whole time. Market Bubble puts a
white lower third under the tiles for as long as a guest is on air:

    LIVE WITH AL DUNLAP
    CEO OF NETNET CAPITAL MANAGEMENT

That is a name, a role and a company, none of which appears anywhere in
the transcript, stated by the show rather than inferred by us. And
because the banner stays up for the whole segment rather than flashing
at the introduction, it also gives the INTERVAL the guest was on -- which
is the thing that makes a voiceprint for them buildable without anyone
labelling audio by hand.

What this does NOT do is say who is speaking. Every tile is captioned at
once and nothing on screen marks the talker: measured over fourteen
seconds of live conversation, the tile borders moved by 1.73 and 0.70
units of brightness and never alternated. So this names the people in
the room and leaves the turns to the audio.

Cost. A per-frame seek costs about fifteen seconds, so sampling a
four-hour show one frame at a time would take longer than the show. The
video is fetched ONCE at low resolution instead, framed locally in a
single ffmpeg pass, and deleted. Frames are only read where a banner is
actually present, which is a pixel test costing nothing.

Writes data/guest_windows.json: episode -> [{name, subtitle, start, end}].
Nothing here touches an embedding or a transcript.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EPISODES = ROOT / "data" / "episodes.json"
BROADCASTS = ROOT / "data" / "broadcast_links.json"
OUT = ROOT / "data" / "guest_windows.json"

YTDLP = "/Users/akhileshmeena/mcg-search/.venv/bin/yt-dlp"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

# One frame every half minute. The banner sits up for the length of a
# guest segment -- five minutes and more, measured -- so this cannot
# step over one, and it keeps a four-hour show to a few hundred frames.
EVERY_SECONDS = 30

# The banner is a solid white bar. A row belonging to it is mostly
# bright, which is a different test from "the brightest row in the
# frame": the plates are white type on a DARK bar and win that contest
# for the wrong reasons.
BRIGHT = 185.0
BANNER_INK = 0.45
BANNER_MIN_H = 30
BANNER_SEARCH_FROM = 0.55
# The left of the banner is the show's logo, not text.
LOGO_FRACTION = 0.18


def log(msg: str) -> None:
    print(msg, flush=True)


# -- reading one frame ---------------------------------------------------

def _runs(mask: np.ndarray, min_h: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_h:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_h:
        out.append((start, len(mask)))
    return out


def banner_runs(a: np.ndarray) -> list[tuple[int, int]]:
    """The bright row-runs of the lower third, unmerged.

    Two runs means two lines: the big name line and the smaller role
    line beneath it. Read separately they survive blur that defeats the
    whole strip -- "LYLER BERNADE" as one strip, "TYLER BERNABE" as one
    line.
    """
    h = a.shape[0]
    ink = (a > BRIGHT).mean(axis=1)
    lo = int(h * BANNER_SEARCH_FROM)
    mask = np.zeros(h, dtype=bool)
    mask[lo:] = ink[lo:] > BANNER_INK
    # Collect first, MERGE, then apply the height minimum. Doing it the
    # other way round was the whole bug: the two text lines of a banner
    # are separated by a 1-2px gap, and anti-aliasing can split them
    # further, so the qualifying rows arrive as several short pieces.
    # Al Dunlap's banner happened to come back as one 57px run and was
    # found; Will Clemente's and Tyler Bernabe's came back as
    # (354,362) (364,388) (389,411) -- 54 qualifying rows in the same
    # place, at the same brightness, every piece under 30px. Both were
    # dropped, and it looked like a resolution or contrast problem for
    # most of a day.
    pieces = _runs(mask, 1)
    if not pieces:
        return []
    gap = max(4, int(h * 0.015))
    merged = [list(pieces[0])]
    for y0, y1 in pieces[1:]:
        if y0 - merged[-1][1] <= gap:
            merged[-1][1] = y1
        else:
            merged.append([y0, y1])
    return [(y0, y1) for y0, y1 in merged if y1 - y0 >= BANNER_MIN_H]


def banner_band(a: np.ndarray) -> tuple[int, int] | None:
    """The white lower third, or None.

    Measured by how much of the row is bright rather than by its mean:
    the banner shares its rows with the dark show logo on the left and a
    dark margin on the right, and those drag a row mean below any
    threshold a white bar should clear.
    """
    h = a.shape[0]
    ink = (a > BRIGHT).mean(axis=1)
    lo = int(h * BANNER_SEARCH_FROM)
    mask = np.zeros(h, dtype=bool)
    mask[lo:] = ink[lo:] > BANNER_INK
    runs = _runs(mask, BANNER_MIN_H)
    if not runs:
        return None
    # Adjacent runs are the two lines of one banner, split by the gap
    # between them. Merge anything close enough to be the same bar.
    merged = [list(runs[0])]
    for y0, y1 in runs[1:]:
        if y0 - merged[-1][1] <= max(6, int(h * 0.02)):
            merged[-1][1] = y1
        else:
            merged.append([y0, y1])
    return tuple(max(merged, key=lambda r: r[1] - r[0]))


def _otsu(a: np.ndarray) -> float:
    hist, _ = np.histogram(a, bins=256, range=(0, 256))
    total, sum_all = a.size, float((np.arange(256) * hist).sum())
    best_thr, best_var, w_b, sum_b = 128.0, -1.0, 0.0, 0.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        var = w_b * w_f * ((sum_b / w_b) - ((sum_all - sum_b) / w_f)) ** 2
        if var > best_var:
            best_var, best_thr = var, float(t)
    return best_thr


def read_banner(im: Image.Image, band: tuple[int, int], psm: int = 3) -> str:
    """OCR the banner. Dark type on white is genuinely bimodal, so the
    split is found rather than fixed -- a fixed cut read this perfectly
    at 720p and returned nothing at 1080p."""
    crop = im.crop((0, band[0], im.width, band[1])).convert("L")
    # Scale to a TARGET height, not by an integer multiple. Rounding
    # 120/height to an int gives 1x for a 69px band and 2x for a 32px
    # one, so the type after scaling lands wherever the arithmetic
    # happens to put it -- which is why this read perfectly at 1280 and
    # 480 and returned nothing at every width in between. Same band,
    # same ink, unreadable type.
    TARGET_H = 150.0
    if crop.height and crop.height < TARGET_H:
        f = TARGET_H / crop.height
        crop = crop.resize((max(1, int(crop.width * f)),
                            max(1, int(crop.height * f))), Image.LANCZOS)
    a = np.asarray(crop, dtype=float)
    bw = Image.fromarray(np.where(a <= _otsu(a), 0, 255).astype(np.uint8))
    # The show's logo sits in a black block at the left of the bar. Drop
    # it: it contributes nothing and it is what psm 6 chokes on.
    bw = bw.crop((int(bw.width * LOGO_FRACTION), 0, bw.width, bw.height))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        bw.save(fh.name)
        # psm 3, not 6. The banner is a 15:1 strip carrying two lines of
        # different sizes, and psm 6 -- "a uniform block of text" --
        # returns an EMPTY STRING on it rather than an error. That is
        # what produced the nonsense where the same banner read at 1280
        # and 480 and vanished at every width in between: nothing to do
        # with resolution, scaling or thresholds, all of which were
        # measured identical. psm 11 reads it too but reorders the
        # lines, putting the role before the name and breaking the
        # "LIVE WITH <name>" parse below.
        got = subprocess.run([shutil.which("tesseract") or "tesseract",
                              fh.name, "stdout", "--psm", str(psm)],
                             capture_output=True, text=True)
    Path(fh.name).unlink(missing_ok=True)
    return " ".join(got.stdout.split())


# "LIVE WITH <name>" names a person. Anything else on that bar is a
# topic card -- "IS GTA DROPPING A COIN?", "WATCH & EARN" -- and names
# nobody, so it must not be mistaken for a guest.
LIVE = re.compile(r"\bL\s*[I1l]\s*V\s*E\s+W[A-Za-z1l|]{1,5}\s+(.+)", re.I)
# Junk the show's own logo leaves at the left edge of the crop.
LEAD_JUNK = re.compile(r"^(?:bubble|market|buble|[^A-Za-z]+)\s*", re.I)


def guest_from(text: str) -> tuple[str, str] | None:
    m = LIVE.search(text)
    if not m:
        return None
    rest = LEAD_JUNK.sub("", m.group(1)).strip()
    rest = re.sub(r"\s+", " ", rest)
    if not rest:
        return None
    # The name runs until the subtitle starts. The subtitle is the role,
    # and it reliably begins with one of these.
    role = re.search(r"\b(CEO|FOUNDER|CO-?FOUNDER|TRADER|CREATOR|INVESTOR|"
                     r"PRESIDENT|HEAD OF|PARTNER|LEADING|ANALYST|BUILDER)\b",
                     rest, re.I)
    if role and role.start() > 0:
        name, subtitle = rest[:role.start()], rest[role.start():]
    else:
        name, subtitle = rest, ""
    name = re.sub(r"[^A-Za-z0-9 .'&-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" .-&")
    subtitle = re.sub(r"\s+", " ", subtitle).strip(" .-&")
    if len(name) < 3:
        return None
    return name.upper(), subtitle.upper()


def _name_quality(name: str) -> int:
    """Whole words score, fragments are penalised. A garbled read is
    full of one- and two-letter pieces; a clean one is not."""
    words = name.split()
    return (sum(len(w) for w in words if len(w) > 2)
            - 2 * sum(1 for w in words if len(w) <= 2))


def read_frame(path: Path) -> tuple[str, str] | None:
    im = Image.open(path).convert("RGB")
    a = np.asarray(im.convert("L"), dtype=float)
    runs = banner_runs(a)
    if not runs:
        return None
    # Whole strip first -- it keeps the role line attached to the name.
    texts = [read_banner(im, (runs[0][0], runs[-1][1]), psm=3)]
    if len(runs) > 1:
        # Then the name line alone, which survives blur the strip does
        # not. Ranking these by subtitle length picked the garbled
        # "LYLER BERNADE" over a clean "TYLER BERNABE", so name quality
        # decides and the subtitle is borrowed from whichever read has
        # one.
        texts.append(read_banner(im, runs[0], psm=7))
    parsed = [g for g in (guest_from(t) for t in texts) if g]
    if not parsed:
        return None
    name = max(parsed, key=lambda g: _name_quality(g[0]))[0]
    subtitle = max((g[1] for g in parsed), key=len, default="")
    return name, subtitle


# -- one episode ---------------------------------------------------------

def video_url(ep: dict, links: dict) -> str | None:
    """Where the pictures are. A YouTube upload is its own url; an X
    show is a status page, and the watchable thing is the broadcast it
    was cut from."""
    eid = ep["episode_id"]
    if not eid.startswith("x-"):
        return ep.get("url")
    return links.get(eid)


def frames_for(url: str, workdir: Path, height: int,
               section: tuple[int, int] | None = None) -> list[tuple[int, Path]]:
    """Fetch once, frame locally. Returns (seconds, path) pairs.

    `section` limits the fetch to (start, end) seconds. A whole
    four-hour broadcast is about 1.7GB at this height and crawls --
    measured at 16MB/min, which is hours. Bounded sections have been
    reliable where full fetches stall, and most of a show is the two
    hosts talking to each other anyway.
    """
    video = workdir / "video.mp4"
    cmd = [YTDLP, "--remote-components", "ejs:github",
           "-f", f"bv*[height<={height}]/b[height<={height}]/worst"]
    if section:
        cmd += ["--download-sections", f"*{section[0]}-{section[1]}",
                "--force-keyframes-at-cuts"]
    cmd += ["-o", str(video), url]
    fetch = subprocess.run(cmd, capture_output=True, text=True)
    if not video.exists():
        err = (fetch.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        log(f"  !! download failed — {err[:150]}")
        return []
    out = workdir / "f_%05d.png"
    subprocess.run(
        [FFMPEG, "-loglevel", "error", "-i", str(video),
         "-vf", f"fps=1/{EVERY_SECONDS}", str(out)],
        capture_output=True, text=True)
    video.unlink(missing_ok=True)
    base = section[0] if section else 0
    frames = sorted(workdir.glob("f_*.png"))
    return [(base + i * EVERY_SECONDS, p) for i, p in enumerate(frames)]


def fill_dropouts(reads: list[tuple[int, tuple[str, str] | None, bool]]
                  ) -> list[tuple[int, tuple[str, str] | None]]:
    """Carry a name across frames where the banner was up but unread.

    The reader is not perfect on every frame: measured across Will
    Clemente's segment, the band was present in all fifteen frames and
    parsed in seven. Treating each frame independently split his window
    in two and cost 19% of his coverage.

    This only fills a frame whose band was actually FOUND -- the bar was
    measurably on screen -- and only when the frames either side name
    the same person. It asserts nothing about a frame with no banner.
    """
    out: list[tuple[int, tuple[str, str] | None]] = []
    for i, (secs, got, had_band) in enumerate(reads):
        if got is None and had_band:
            before = next((g for _, g, _ in reversed(reads[:i]) if g), None)
            after = next((g for _, g, _ in reads[i + 1:] if g), None)
            if before and after and before[0] == after[0]:
                got = before
        out.append((secs, got))
    return out


def windows_from(reads: list[tuple[int, tuple[str, str] | None]]) -> list[dict]:
    """Contiguous frames naming the same person become one window.

    The end is the last frame the banner was still up, plus the sampling
    step -- the guest was there for that gap too, we simply did not look.
    """
    out: list[dict] = []
    for secs, got in reads:
        if got is None:
            continue
        name, subtitle = got
        if out and out[-1]["name"] == name and \
                secs - out[-1]["_last"] <= EVERY_SECONDS * 3:
            out[-1]["_last"] = secs
            out[-1]["end"] = secs + EVERY_SECONDS
            if subtitle and not out[-1]["subtitle"]:
                out[-1]["subtitle"] = subtitle
        else:
            out.append({"name": name, "subtitle": subtitle,
                        "start": secs, "end": secs + EVERY_SECONDS,
                        "_last": secs})
    for w in out:
        w.pop("_last", None)
    return out


def main(argv: list[str]) -> int:
    only = None
    keep = "--keep" in argv
    height = 720
    if "--only" in argv:
        only = argv[argv.index("--only") + 1]
    if "--height" in argv:
        height = int(argv[argv.index("--height") + 1])
    section = None
    if "--from" in argv and "--to" in argv:
        section = (int(argv[argv.index("--from") + 1]),
                   int(argv[argv.index("--to") + 1]))

    episodes = json.loads(EPISODES.read_text())
    links = json.loads(BROADCASTS.read_text()) if BROADCASTS.exists() else {}
    if only:
        episodes = [e for e in episodes if e["episode_id"] == only]
        if not episodes:
            log(f"no episode {only}")
            return 1

    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    for ep in episodes:
        eid = ep["episode_id"]
        if eid in done and not only:
            continue
        url = video_url(ep, links)
        if not url:
            log(f"{eid}: no video url — skipped")
            continue
        log(f"\n{eid}  {ep['title'][:60]}")
        work = Path(tempfile.mkdtemp(prefix=f"gw_{eid}_"))
        try:
            frames = frames_for(url, work, height, section)
            if not frames:
                continue
            log(f"  {len(frames)} frames @ {EVERY_SECONDS}s, {height}p")
            raw = []
            for secs, path in frames:
                arr = np.asarray(Image.open(path).convert("L"), dtype=float)
                raw.append((secs, read_frame(path), bool(banner_runs(arr))))
            reads = fill_dropouts(raw)
            found = windows_from(reads)
            if section and eid in done:
                # A sectioned run only saw part of the show. Replacing
                # the episode's entry would throw away windows found by
                # an earlier section -- keep anything outside this one.
                kept = [w for w in done[eid]
                        if w["end"] <= section[0] or w["start"] >= section[1]]
                found = sorted(kept + found, key=lambda w: w["start"])
            done[eid] = found
            # Save per episode: a run over the archive must not lose an
            # hour of work to one bad download at the end.
            OUT.write_text(json.dumps(done, indent=1, ensure_ascii=False))
            for w in found:
                a, b = w["start"], w["end"]
                log(f"    {a//3600}:{a%3600//60:02d}:{a%60:02d}"
                    f"-{b//3600}:{b%3600//60:02d}:{b%60:02d}  "
                    f"{w['name']}  |  {w['subtitle'][:44]}")
            if not found:
                log("    (no guest banner found)")
        finally:
            if not keep:
                shutil.rmtree(work, ignore_errors=True)
            else:
                log(f"  frames kept in {work}")
    log(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
