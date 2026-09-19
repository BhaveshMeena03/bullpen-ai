"""Map each second of a live X broadcast to the same second on YouTube.

    .venv/bin/python scripts/build_youtube_map.py
    .venv/bin/python scripts/build_youtube_map.py --report

Most shows are indexed twice: the full live broadcast on X and the
YouTube upload cut from it a day later. A moment in both should play on
YouTube, because YouTube plays inside the page and X opens another site.
But the two run on different clocks -- the upload drops the waiting
screen, the pre-show and about a quarter of the rest, in pieces -- so a
single offset is wrong for most of the show. align_recordings.py finds
that one offset; this finds it for every moment.

Eight-word runs, as there: the copies were transcribed separately, so
whole lines rarely match, but runs of eight words do, and a run that
occurs once in each copy pins the same instant in both. Each live
segment takes the YouTube second its runs agree on, and anchors that
disagree with their neighbours are dropped -- a cut is a jump in the
offset, not noise to average over.

Writes data/youtube_map.json: live episode -> [[live_s, youtube_id,
youtube_s], ...], about one anchor every ten seconds of matched speech.
A second with no anchor near it was cut from the upload, and stays on X.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import episode_store  # noqa: E402
from app.dedupe import dedupe, group_by_show  # noqa: E402

OUT = ROOT / "data" / "youtube_map.json"
RUN = 8                 # words per run
EVERY = 10.0            # seconds between kept anchors
AGREE = 25.0            # an anchor must sit within this of its neighbours' offset

_WORD = re.compile(r"[a-z0-9']+")


def _words(episode: dict) -> list[tuple[str, float]]:
    """Every word, with the start second of the line it is in."""
    out = []
    for seg in sorted(episode["segments"], key=lambda s: s.get("t", 0)):
        for w in _WORD.findall((seg.get("text") or "").lower()):
            out.append((w, float(seg.get("t", 0))))
    return out


def _runs(words: list[tuple[str, float]]) -> dict[str, list[float]]:
    runs: dict[str, list[float]] = collections.defaultdict(list)
    for i in range(len(words) - RUN + 1):
        runs[" ".join(w for w, _ in words[i:i + RUN])].append(words[i][1])
    return runs


def anchors(live: dict, cuts: list[dict]) -> list[list]:
    """[live_s, youtube_id, youtube_s] where both copies say the same run."""
    unique: dict[str, tuple[str, float]] = {}
    seen: collections.Counter = collections.Counter()
    for cut in cuts:
        for run, times in _runs(_words(cut)).items():
            seen[run] += len(times)
            unique[run] = (cut["episode_id"], times[0])
    raw = []
    for run, times in _runs(_words(live)).items():
        # Once in each copy, or it pins nothing: "we'll be right back"
        # recurs, and a run that recurs could be either occurrence.
        if len(times) == 1 and seen[run] == 1:
            yt_id, yt_s = unique[run]
            raw.append((times[0], yt_id, yt_s))
    raw.sort()

    # Keep an anchor only if the ones around it tell the same story. A
    # stray match across the show would otherwise send a click to the
    # wrong minute; a real cut shows up as a run of anchors that all
    # jump together.
    kept = []
    for i, (live_s, yt_id, yt_s) in enumerate(raw):
        near = [r for r in raw[max(0, i - 6):i + 7] if r[1] == yt_id]
        offsets = sorted(r[2] - r[0] for r in near)
        median = offsets[len(offsets) // 2]
        if abs((yt_s - live_s) - median) <= AGREE and len(near) >= 5:
            if not kept or live_s - kept[-1][0] >= EVERY:
                kept.append([round(live_s, 1), yt_id, round(yt_s, 1)])
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="measure, write nothing")
    args = ap.parse_args()

    kept, _ = dedupe(episode_store.load())
    mapping, lines = {}, []
    for group in group_by_show(kept):
        lives = [e for e in group if e["episode_id"].startswith("x-")]
        cuts = [e for e in group if not e["episode_id"].startswith("x-")]
        for live in lives:
            if not cuts:
                continue
            found = anchors(live, cuts)
            if not found:
                continue
            mapping[live["episode_id"]] = found
            length = max(s.get("t", 0) for s in live["segments"]) or 1
            # Coverage: seconds of the live show within one anchor gap of
            # an anchor, which is what a click can actually be mapped from.
            covered = sum(min(EVERY * 2, b[0] - a[0])
                          for a, b in zip(found, found[1:]))
            lines.append(f"  {live.get('published_at', '')[:10]}  "
                         f"{len(found):4} anchors  {covered / length:4.0%} of the "
                         f"live show maps to YouTube  {live['title'][:44]}")
    print("\n".join(sorted(lines)))
    print(f"\n  {len(mapping)} live broadcasts mapped")
    if not args.report:
        OUT.write_text(json.dumps(mapping, separators=(",", ":")))
        print(f"  wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
