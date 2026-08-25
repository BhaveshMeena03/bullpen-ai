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

One thing is worse, and it is the important one. X has no timestamp
parameter for video, so a citation cannot deep-link to the moment the way a
YouTube one does. The answer still names the timestamp; the viewer has to
scrub to it. That is a real downgrade of the best feature here, and the
reason these are marked platform="other" rather than quietly presented as
equivalent.

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


def _norm(text: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ",
                   _re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def overlap_with_existing(segments: list[dict], existing: list[dict]) -> tuple[float, str]:
    """How much of this transcript is already indexed, and where.

    Added after nearly ingesting a 59-minute Orangie interview that turned
    out to be 62% contained in the YouTube upload of episode #16. Two
    copies of the same words is not merely wasteful: the duplicate competes
    for the same retrieval slots, so a search returns one moment twice
    instead of two moments — and it can cite the X copy, which has no
    timestamp deep-link, when YouTube has the identical passage WITH one.

    Sampled rather than exhaustive; this only needs to tell "basically the
    same recording" from "genuinely new".
    """
    import random
    mine = _norm(" ".join(s["text"] for s in segments))
    words = mine.split()
    if len(words) < 200:
        return 0.0, ""
    random.seed(7)          # deterministic, so re-runs agree with each other
    starts = random.sample(range(0, len(words) - 10), min(300, len(words) - 10))
    best, where = 0.0, ""
    for ep in existing:
        other = _norm(" ".join(s["text"] for s in ep["segments"]))
        hits = sum(1 for i in starts if " ".join(words[i:i + 7]) in other)
        ratio = hits / len(starts)
        if ratio > best:
            best, where = ratio, f"{ep['episode_id']} ({ep['title'][:40]})"
    return best, where


def build(url: str) -> dict | None:
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
    published = (datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                 if ts else None)
    return {
        # Prefixed so an X segment can never collide with a YouTube id, and
        # so it is obvious in logs and citations where a hit came from.
        "episode_id": f"x-{status_id}",
        "title": title_from(meta),
        "url": f"https://x.com/{handle}/status/{status_id}",
        # Not "youtube": the deep-link builder must not append ?t=, which X
        # ignores, producing a link that silently lands at 0:00.
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
    ap.add_argument("--max-overlap", type=float, default=0.35,
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
    added = 0
    for url in urls:
        ep = build(url)
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
        added += 1
        mins = (ep["segments"][-1]["t"] / 60) if ep["segments"] else 0
        print(f"  {state:8s} {ep['title'][:48]:50s} "
              f"{len(ep['segments']):5d} cues  {mins:5.1f} min")

    merged = sorted(by_id.values(),
                    key=lambda e: (e.get("published_at") or ""), reverse=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False))
    print(f"\n{added} written · {len(merged)} episodes total -> "
          f"{out_path.relative_to(ROOT)}")
    print("Now re-ingest:  .venv/bin/python scripts/ingest_episodes.py")


if __name__ == "__main__":
    main()
