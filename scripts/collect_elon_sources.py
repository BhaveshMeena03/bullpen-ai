"""Build a reviewable list of Elon Musk sources worth transcribing.

    .venv/bin/python scripts/collect_elon_sources.py
    .venv/bin/python scripts/collect_elon_sources.py --out /tmp/elon.json

Writes a list. Ingests nothing, on purpose: transcription is hours of
laptop time per source and the decision of what belongs in the archive is
the whole product, so it is made by a person looking at a list.

The bar is higher here than it was for Market Bubble, for one reason. That
show has one broadcaster and the audio is unambiguous. Elon Musk has an
enormous volume of misattributed quotes, re-cut compilations and outright
synthetic audio in circulation, and an archive that cites a fake as
precisely as it cites a real earnings call is worse than no archive. So
provenance decides what is listed:

  official     Tesla's own channel. Quarterly, unambiguously dated, and
               nobody else's edit of what was said. This is the spine.
  interview    a handful of long-form conversations, on the channel that
               recorded them. Listed with the channel visible so a person
               can reject anything that is not the original publisher.

Everything else -- compilations, "motivational" cuts, reaction channels,
anything on a channel that is not the one that made the recording -- is
left off. Not because it is necessarily wrong, but because verifying it
costs more than re-transcribing the original.

Dates come from a second pass. --flat-playlist enumerates a channel
cheaply and returns no upload date at all, which is why an earlier version
of this listed everything as NA; the real date is fetched only for the
items that survive the filter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

YTDLP = str(ROOT / ".venv" / "bin" / "yt-dlp")
if not Path(YTDLP).exists():
    YTDLP = "yt-dlp"

# Enumerated in full. The earnings calls sit behind a wall of short
# marketing clips -- an earlier run stopped at 60 videos and concluded
# Tesla had never posted one.
# Tesla curates its own earnings calls into a playlist, which is a better
# source than the channel feed for the same reason a table of contents
# beats a shelf: it is complete, it is theirs, and it does not stop.
#
# The channel feed does stop. YouTube returns about 375 videos for
# @tesla/videos however deep the request goes, the calls are older than
# that window, and an earlier run of this concluded from the feed that
# Tesla had posted exactly one earnings call in its history. The playlist
# holds 21, every quarter from Q2 2021 to Q2 2026.
EARNINGS_PLAYLIST = ("Tesla earnings calls",
                     "https://www.youtube.com/playlist?"
                     "list=PLEox0nUMFPF4TOgwz6PYqoV_lhyENbWr6", 100)
OFFICIAL = [EARNINGS_PLAYLIST,
            ("Tesla channel", "https://www.youtube.com/@tesla/videos", 400)]

# What an official long-form recording is called. Matched on the title
# because the channel posts both these and 60-second adverts.
ON_THE_RECORD = ("financial results", "earnings call", "q&a webcast",
                 "shareholder", "annual meeting", "investor day")

# Searched rather than listed, because they live on other people's
# channels. Every hit is reported with its channel so a person can throw
# out anything that is not the original publisher.
SEARCHES = [
    "Elon Musk Lex Fridman podcast full episode",
    "Elon Musk Joe Rogan Experience full episode",
    "Elon Musk All-In podcast full interview",
    "Elon Musk TED interview full",
    "SpaceX Starship update Elon Musk presentation full",
]

# Under twenty minutes is a clip of something, not the thing itself.
LONG_ENOUGH = 20 * 60


def enumerate_flat(target: str, limit: int) -> list[dict]:
    """id, duration, channel and title, without paying for per-video pages."""
    out: list[dict] = []
    fields = "%(id)s\t%(duration)s\t%(channel)s\t%(title)s"
    proc = subprocess.run(
        [YTDLP, "--no-warnings", "--flat-playlist", "--playlist-end",
         str(limit), "--print", fields, target],
        capture_output=True, text=True, timeout=600)
    for line in (proc.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            seconds = int(float(parts[1]))
        except (TypeError, ValueError):
            seconds = 0
        out.append({"id": parts[0], "seconds": seconds,
                    "channel": parts[2], "title": parts[3]})
    return out


def real_date(video_id: str) -> str:
    """The upload date, which --flat-playlist does not carry.

    One page fetch per surviving item, which is why it runs after the
    filter and not before it.
    """
    proc = subprocess.run(
        [YTDLP, "--no-warnings", "--skip-download", "--print",
         "%(upload_date)s", f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, text=True, timeout=120)
    stamp = (proc.stdout or "").strip().splitlines()
    value = stamp[-1] if stamp else ""
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/elon_sources.json"))
    ap.add_argument("--skip-dates", action="store_true",
                    help="faster; leaves every date blank")
    args = ap.parse_args()

    found: dict[str, dict] = {}

    for label, url, limit in OFFICIAL:
        print(f"  enumerating {label} ({limit} videos)…", flush=True)
        for item in enumerate_flat(url, limit):
            title = item["title"].lower()
            if any(mark in title for mark in ON_THE_RECORD):
                item["source"] = "official"
                found[item["id"]] = item

    for query in SEARCHES:
        print(f"  searching: {query}", flush=True)
        for item in enumerate_flat(f"ytsearch6:{query}", 6):
            if item["seconds"] < LONG_ENOUGH:
                continue
            item["source"] = "interview (check the channel)"
            found.setdefault(item["id"], item)

    items = list(found.values())
    if not args.skip_dates:
        print(f"  dating {len(items)} candidates…", flush=True)
        for item in items:
            item["date"] = real_date(item["id"])

    items.sort(key=lambda x: (x.get("source", ""), x.get("date") or ""))
    args.out.write_text(json.dumps(items, indent=1))

    hours = sum(i["seconds"] for i in items) / 3600
    official = [i for i in items if i["source"] == "official"]
    print(f"\n  {len(items)} candidates, {hours:.0f}h total")
    print(f"  {len(official)} official "
          f"({sum(i['seconds'] for i in official)/3600:.0f}h), "
          f"{len(items) - len(official)} to check by hand")
    print(f"  -> {args.out}\n")
    for item in items:
        print(f"  {item.get('date','') or '????-??-??'}  "
              f"{item['seconds']/60:5.0f}m  {item['channel'][:18]:20}"
              f"{item['title'][:52]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
