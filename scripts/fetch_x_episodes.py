"""Pull Market Bubble broadcast segments from X into the Episode JSON.

    .venv/bin/python scripts/fetch_x_episodes.py URL [URL ...]
    .venv/bin/python scripts/fetch_x_episodes.py --urls data/x_broadcasts.txt

Why this exists: the guest interviews are not on YouTube. Three found so far
— Orangie, Austin Federa of DoubleZero, and the Squire founder — all went out
live and were posted to X, while the channel only carries the main show. A
viewer asked the search about Squire, got nothing, and reasonably concluded
the tool was broken. It was not broken; it could not see half the catalogue.

Two things turned out better than expected:

  captions   X serves a real English subtitle track for these, so nothing
             needs transcribing. They are also cleaner than YouTube's
             auto-captions — actual punctuation and capitalisation, rather
             than a run-on smear — which makes the chunks read better and
             the quotes citable verbatim.
  auth       No login or cookies needed for public posts.

One thing is worse, though less than it looks. A citation into a broadcast
does land on the moment — ?t=<seconds> seeks on the status URL, checked on
three broadcasts — so these are marked platform="other" only to keep the
"s" suffix off the parameter, which X does not accept. What X lacks is an
embed: the moment opens on X rather than inside the page, which is why
_prefer_seekable in app/podcast.py still favours a YouTube copy of the
same words.

There is no way to enumerate an account's videos — yt-dlp rejects a bare
profile URL — so the URLs are supplied by hand, one per line in a file.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.captions import parse_vtt  # noqa: E402
from app.dedupe import SAME_RECORDING, overlap  # noqa: E402
from app.episode_store import merge as merge_episodes  # noqa: E402

OUT = ROOT / "data" / "episodes.json"

_STATUS = re.compile(r"(?:twitter|x)\.com/([^/]+)/status/(\d+)")


def _ytdlp() -> str:
    local = ROOT / ".venv" / "bin" / "yt-dlp"
    return str(local) if local.exists() else (shutil.which("yt-dlp") or "yt-dlp")


def probe(url: str) -> dict | None:
    """Metadata for one post, or None if it has no readable video."""
    try:
        raw = subprocess.run(
            [_ytdlp(), "--socket-timeout", "30", "-J", "--no-warnings", url],
            capture_output=True, text=True, timeout=180, check=True,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"  SKIP {url}: {exc}")
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  SKIP {url}: metadata was not JSON")
        return None


def captions(url: str) -> str | None:
    """The English subtitle track, or None when the post has none."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sub"
        try:
            subprocess.run(
                [_ytdlp(), "--socket-timeout", "30", "--skip-download",
                 "--write-subs", "--sub-langs", "en", "--sub-format", "vtt",
                 "-o", str(out) + ".%(ext)s", url],
                capture_output=True, text=True, timeout=600, check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"  SKIP {url}: subtitle download failed ({exc})")
            return None
        files = list(Path(tmp).glob("*.vtt"))
        return files[0].read_text(errors="replace") if files else None


def title_from(meta: dict) -> str:
    """A readable episode title from the post text.

    The post usually opens with a sentence naming the guest ("Our full
    conversation with Orangie.") and then lists chapter timestamps. Only the
    first sentence is a title; the chapter list is content and would make an
    unreadable one.
    """
    raw = (meta.get("description") or meta.get("title") or "").strip()
    raw = re.sub(r"https?://\S+", "", raw)
    first = re.split(r"(?:\n|(?<=[.!?])\s)", raw.strip(), maxsplit=1)[0].strip()
    first = re.sub(r"\s+", " ", first)
    if 8 <= len(first) <= 120:
        return first
    return (meta.get("uploader") or "Market Bubble") + " broadcast"


def overlap_with_existing(segments: list[dict],
                          existing: list[dict]) -> tuple[float, str]:
    """How much of this transcript is already indexed, and where.

    Thin wrapper over app.dedupe so there is one definition of "already
    indexed" in the codebase. Having had two, with different thresholds,
    is how a 62%-overlapping interview got refused while an equally
    overlapping pair sat happily in the index.
    """
    best, where = 0.0, ""
    for ep in existing:
        ratio = overlap(segments, ep.get("segments") or [])
        if ratio > best:
            best, where = ratio, f"{ep['episode_id']} ({ep['title'][:40]})"
    return best, where


def build(url: str, recorded: str | None = None) -> dict | None:
    meta = probe(url)
    if meta is None:
        return None
    m = _STATUS.search(url)
    if not m:
        print(f"  SKIP {url}: not an x.com/<user>/status/<id> URL")
        return None
    handle, status_id = m.groups()

    vtt = captions(url)
    if not vtt:
        print(f"  SKIP {url}: no English captions — would need transcribing")
        return None
    segments = parse_vtt(vtt)
    if len(segments) < 20:
        print(f"  SKIP {url}: only {len(segments)} caption cues, looks empty")
        return None

    ts = meta.get("timestamp")
    published = recorded or (datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                             if ts else None)
    return {
        # Prefixed so an X segment can never collide with a YouTube id, and
        # so it is obvious in logs and citations where a hit came from.
        "episode_id": f"x-{status_id}",
        "title": title_from(meta),
        "url": f"https://x.com/{handle}/status/{status_id}",
        # Not "youtube": X seeks on ?t=<seconds> but rejects the trailing
        # "s" that YouTube requires, and the deep-link builder keys the
        # suffix off this field.
        "platform": "other",
        "published_at": published,
        "segments": segments,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="*", help="x.com status URLs")
    ap.add_argument("--urls", dest="url_file",
                    help="file with one URL per line (# comments allowed)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--date", metavar="YYYY-MM-DD",
                    help="the date this was RECORDED, when it differs from "
                         "when it was posted. A live broadcast is posted the "
                         "day it airs, so its post date is right. A clip is "
                         "posted whenever someone got around to cutting it, "
                         "and taking that as the air date makes an old "
                         "conversation look like the newest thing in the "
                         "index — which matters, because the answers reason "
                         "about recency ('what does he think now').")
    ap.add_argument("--max-overlap", type=float, default=SAME_RECORDING,
                    help="refuse a segment already this contained in an "
                         "indexed episode (default 0.35)")
    args = ap.parse_args()

    urls = list(args.urls)
    if args.url_file:
        for line in Path(args.url_file).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                urls.append(line)
    if not urls:
        sys.exit("no URLs given")

    out_path = Path(args.out)
    existing = (json.loads(out_path.read_text())
                if out_path.exists() else [])
    by_id = {e["episode_id"]: e for e in existing}

    print(f"  {len(urls)} URL(s), {len(existing)} episode(s) already on file\n")
    # Only what THIS run produced gets written back. Everything else in
    # by_id is a stale snapshot read at startup, and re-writing it is the
    # very thing that used to revert other processes' work.
    written_ids: set[str] = set()
    added = 0
    for url in urls:
        ep = build(url, args.date)
        if ep is None:
            continue
        others = [e for e in by_id.values() if e["episode_id"] != ep["episode_id"]]
        ratio, where = overlap_with_existing(ep["segments"], others)
        if ratio >= args.max_overlap:
            print(f"  SKIP     {ep['title'][:44]:46s} "
                  f"{ratio:.0%} already in {where}")
            print("           (pass --max-overlap 1.0 to add it anyway)")
            continue
        if ratio > 0.15:
            print(f"  note     {ratio:.0%} of this also appears in {where}")
        # Re-running replaces rather than duplicates, so a corrected caption
        # track can simply be fetched again.
        state = "updated" if ep["episode_id"] in by_id else "added"
        by_id[ep["episode_id"]] = ep
        written_ids.add(ep["episode_id"])
        added += 1
        mins = (ep["segments"][-1]["t"] / 60) if ep["segments"] else 0
        print(f"  {state:8s} {ep['title'][:48]:50s} "
              f"{len(ep['segments']):5d} cues  {mins:5.1f} min")

    # Merged under a lock, re-reading inside it. A run that started before
    # another finished must not revert that other's work — which is exactly
    # how an X clip fetched mid-transcription once vanished.
    new = [e for e in by_id.values() if e["episode_id"] in written_ids]
    merge_episodes(new, out_path)
    total = len(json.loads(out_path.read_text()))
    print(f"\n{added} written · {total} episodes total -> "
          f"{out_path.relative_to(ROOT)}")
    print("Now re-ingest:  .venv/bin/python scripts/ingest_episodes.py")


if __name__ == "__main__":
    main()
