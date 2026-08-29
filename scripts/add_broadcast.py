"""Take a live broadcast from a URL to searchable, in one command.

    .venv/bin/python scripts/add_broadcast.py https://x.com/MarketBubble/status/...
    .venv/bin/python scripts/add_broadcast.py URL --date 2026-09-03

Roughly half of every show is only on X. The episode goes out live, and the
YouTube upload lands a day later with about a third of it cut — so the window
where this index has something nobody else does is the day in between, and it
is worth being able to use that window without remembering four commands in
the right order.

What it does:

    transcribe   locally, on this machine, with MLX Whisper
    ingest       appends to the index — does NOT clear and rebuild
    summarize    so the episode appears when browsing, not only in search
    verify       drops summary timestamps that point at the wrong moment
    terms        rebuilds the exact-token index, so names in the new
                 episode are findable by name and not only by meaning
    highlights   refreshes the pool the bot draws unprompted answers from
    verify       asks the live index a question about it and shows the answer

Appending matters. scripts/ingest_episodes.py clears the namespace and
re-embeds all 2,538 windows to add one episode, which costs real money and
leaves the search returning nothing while it runs. This adds only the new
one.

The YouTube version needs nothing from you: a GitHub Action checks twice a
day and emails when it finds one, and scripts/sync_latest.py finishes the
job. Citations move to YouTube on their own wherever both cover the same
moment, because a YouTube link can jump to the second and an X one cannot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.episode_store import load  # noqa: E402
from app.podcast import PodcastIndex  # noqa: E402
from app.schemas import Episode  # noqa: E402
from app.summaries import SummaryStore  # noqa: E402
from app.x_api import API, XCredentials  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"
SPEAKER_MAP = ROOT / "data" / "speaker_map.json"


def step(n: int, of: int, what: str) -> None:
    print(f"\n{'─' * 62}\n  {n}/{of}  {what}\n{'─' * 62}")


async def still_running(url: str) -> tuple[bool, str]:
    """Is this post still a live stream rather than a finished recording?

    The broadcast post exists from the first minute of a four hour show,
    so a URL pasted the moment it appears downloads only what has aired.
    Twenty minutes of local transcription then produces a partial episode
    that enters the index looking exactly like a whole one, and nothing
    downstream ever re-checks it.

    One read against the post answers it. Any failure here returns "fine"
    — a check that cannot run must not stand between somebody and their
    own archive.
    """
    found = re.search(r"/status/(\d+)", url) or re.search(r"(\d{15,})", url)
    if not found:
        return False, ""
    try:
        settings = get_settings()
        cred = XCredentials(settings.x_api_key, settings.x_api_secret,
                            settings.x_access_token, settings.x_access_secret)
        endpoint = f"{API}/tweets"
        params = {"ids": found.group(1),
                  "tweet.fields": "created_at,attachments",
                  "expansions": "attachments.media_keys",
                  "media.fields": "type,duration_ms"}
        async with httpx.AsyncClient(timeout=20) as http:
            response = await http.get(
                endpoint, params=params,
                headers={"Authorization": cred.header("GET", endpoint, params)})
        if response.status_code != 200:
            return False, ""
        media = (response.json().get("includes", {}).get("media") or [])
        longest = max((m.get("duration_ms") or 0 for m in media), default=0)
    except Exception:                                           # noqa: BLE001
        return False, ""

    if not longest:
        return False, ""
    hours = longest / 3_600_000
    if hours < 2:
        return True, (f"X reports this video as {hours * 60:.0f} minutes "
                      f"long. Every full show has run three to four hours, "
                      f"so this is either still streaming or one of the "
                      f"shorter cut-downs.")
    return False, ""


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="the x.com broadcast or clip URL")
    ap.add_argument("--date", help="YYYY-MM-DD if X reports the wrong one")
    ap.add_argument("--model", default="turbo", choices=["turbo", "large-v3"])
    ap.add_argument("--skip-highlights", action="store_true",
                    help="leave the unprompted-answer pool as it is")
    ap.add_argument("--force", action="store_true",
                    help="index it even if it looks like it is still live")
    args = ap.parse_args()

    partial, why = await still_running(args.url)
    if partial and not args.force:
        print(f"\n  Not starting.\n\n  {why}\n")
        print("  Transcribing now would spend twenty minutes and store a "
              "partial\n  episode that reads like a complete one. Wait until "
              "the stream has\n  ended, or pass --force if you know better.\n")
        return 1
    if partial:
        print(f"\n  --force: {why}\n")

    began = time.time()
    before = {e["episode_id"] for e in load(EPISODES)}

    # 1 — transcribe ---------------------------------------------------------
    step(1, 8, "transcribing (local, no API — this is the slow part)")
    cmd = [sys.executable, "scripts/transcribe_x_broadcast.py", args.url,
           "--model", args.model, "--keep-audio"]
    if args.date:
        cmd += ["--date", args.date]
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        print("\n  transcription failed — nothing was indexed")
        return 1

    episodes = load(EPISODES)
    fresh = [e for e in episodes if e["episode_id"] not in before]
    if not fresh:
        # The transcriber skips a broadcast that overlaps one already held,
        # which is a correct outcome and not a failure.
        print("\n  nothing new — already indexed, or skipped as a duplicate")
        return 0

    # 2 — ingest -------------------------------------------------------------
    step(2, 8, f"indexing {len(fresh)} episode(s) — appending, not rebuilding")
    index = PodcastIndex()
    parsed = [Episode(**e) for e in fresh]
    windows = await index.ingest(parsed)
    print(f"  {windows} searchable passages added")

    # 3 — who is speaking ----------------------------------------------------
    # Before the summary, because the summary reads the transcript and a
    # transcript with names in it produces "Ansem argued X and Banks pushed
    # back" instead of "the hosts discussed X". Never fatal: an episode
    # without labels is the episode everything before tonight had.
    step(3, 8, "working out who is speaking")
    speaker_map: dict[str, dict[str, str]] = {}
    for episode in parsed:
        audio = ROOT / "audio" / f"{episode.episode_id}.mp3"
        if not audio.exists():
            print(f"  no audio kept for {episode.episode_id} — skipping "
                  f"labels, everything else still runs")
            continue
        for label, cmd in (
            ("fingerprinting", [sys.executable, "scripts/label_speakers.py",
                                "--only", episode.episode_id]),
            ("matching against the hosts", [sys.executable,
                                            "scripts/build_speaker_map.py"]),
            ("labelling the passages", [sys.executable,
                                        "scripts/apply_speaker_labels.py",
                                        "--only", episode.episode_id]),
        ):
            print(f"  {label}…", flush=True)
            if subprocess.run(cmd, cwd=ROOT).returncode != 0:
                print(f"  {label} failed — continuing without labels")
                break
    if SPEAKER_MAP.exists():
        try:
            speaker_map = json.loads(SPEAKER_MAP.read_text())
        except json.JSONDecodeError:
            speaker_map = {}

    # 4 — summarize ----------------------------------------------------------
    step(4, 8, "summarising, so it shows up when browsing")
    summaries = SummaryStore()
    for episode in parsed:
        try:
            await summaries.store(episode, await summaries.summarize(
                episode, speakers=speaker_map.get(episode.episode_id)))
            print(f"  summarised {episode.title[:52]}")
        except Exception as exc:                              # noqa: BLE001
            # Not fatal: the episode is already searchable, which is the
            # part that mattered.
            print(f"  summary failed ({exc}) — searchable anyway, rerun later")

    # 4 — check the summary's timestamps -------------------------------------
    # Three of the fourteen topic lines in the first broadcast added after
    # this pipeline existed pointed at the wrong moment — a real transcript
    # marker attached to the wrong topic. They were caught only because
    # somebody was about to quote the summary at the show's host. This runs
    # every time now, because the summary is most of what "summarize the
    # latest episode" returns and nobody reads it before it goes out.
    step(5, 8, "checking the summary's timestamps against the transcript")
    for episode in parsed:
        subprocess.run([sys.executable, "scripts/verify_summaries.py",
                        "--episode-id", episode.episode_id, "--apply"],
                       cwd=ROOT)

    # 5 — the exact-token index ----------------------------------------------
    # Rebuilt here rather than remembered later: a stale index simply has no
    # entry for the new episode, so a question about it silently loses the
    # exact-name matching and nobody finds out.
    step(6, 8, "rebuilding the exact-token index")
    subprocess.run([sys.executable, "scripts/build_term_index.py"], cwd=ROOT)

    # 5 — highlights ---------------------------------------------------------
    step(7, 8, "refreshing the pool the bot answers from unprompted")
    if args.skip_highlights:
        print("  skipped")
    else:
        # Only the new episodes, appended. A bare rebuild re-derives the
        # whole pool and discards every curated decision in it.
        only = []
        for episode in parsed:
            only += ["--only", episode.episode_id]
        subprocess.run([sys.executable, "scripts/make_highlights.py", *only],
                       cwd=ROOT)

    # 6 — prove it -----------------------------------------------------------
    step(8, 8, "asking the live index about it")
    for episode in parsed:
        # Named guests are what people actually search for, and the title is
        # where they are named.
        who = episode.title.split(":")[0].replace("LIVE W/", "").strip()
        question = f"what did {who.split('&')[0].strip().lower()} say"
        result = await index.search(question)
        print(f"\n  Q: {question}")
        print(f"  A: {result.answer[:200]}")
        found = any(h.episode_id == episode.episode_id for h in result.hits)
        print(f"     {'cites the new episode' if found else 'NOT cited yet — '
                     'reranking may just prefer an older take on the same topic'}")

    total = len(load(EPISODES))
    print(f"\n{'─' * 62}")
    print(f"  done in {(time.time() - began) / 60:.0f} min · "
          f"{total} episodes indexed")
    print("  the bot needs no restart — it reads the same index.")
    print(f"{'─' * 62}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
