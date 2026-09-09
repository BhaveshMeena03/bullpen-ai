"""Turn a finished episode into clips somebody can post that morning.

    .venv/bin/python scripts/clip_episode.py                    # newest
    .venv/bin/python scripts/clip_episode.py --episode UIcjWPG7eJ8
    .venv/bin/python scripts/clip_episode.py --count 5 --dry-run

The bottleneck in this kind of show is not making the show. It is that
four hours of it goes up and nobody has time to find the ninety seconds
worth sending to anyone. That work is manual, it is slow, and it is the
reason most of a broadcast is never seen again.

Both halves of doing it already existed here and had never been joined:
make_highlights.py knows how to pick a moment out of a transcript, and
make_clip.py knows how to cut one. This runs the first, then the second on
each result, and leaves a folder of finished files with a draft caption
beside each — the state where the only thing left is a human deciding
whether to post it.

Deliberately stops at "ready to post". Nothing here uploads anything. The
account is a person's, the judgement about what goes out is theirs, and a
script that posts on its own turns one bad pick into a public one.

--dry-run prints the moments and cuts nothing, which is the cheap way to
see whether an episode was worth the encoding time. Cutting is about a
minute per clip on this laptop; choosing is a few cents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anthropic import AsyncAnthropic  # noqa: E402

from app.config import anthropic_client_kwargs, get_settings  # noqa: E402

# Imported rather than reimplemented. A second copy of the moment-picking
# prompt would drift from the first the week nobody remembered it existed.
sys.path.insert(0, str(ROOT / "scripts"))
from make_highlights import highlights_for, safe_to_post  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"
CLIPPER = ROOT / "scripts" / "make_clip.py"

# How many times to ask for moments before giving up. See the loop in
# main(): the picker returns a different number every run, sometimes zero.
ATTEMPTS = 3


def newest(episodes: list[dict]) -> dict:
    """The most recently published episode that has a transcript.

    Sorted on published_at rather than file order: episodes.json is
    appended to by whichever ingest ran last, and the last line in it has
    more than once been a re-run of something from May.
    """
    with_dates = [e for e in episodes if e.get("published_at")]
    if not with_dates:
        raise SystemExit("  no episode carries a published_at")
    return max(with_dates, key=lambda e: e["published_at"])


def caption(moment: dict, episode: dict) -> str:
    """A draft to edit, not a post to publish.

    Lowercase and plain, because that is how the account writes, and the
    fastest draft to fix is one already in the right voice. The timestamp
    and title go in because they are the part a human would otherwise have
    to look up again.
    """
    return (f"{moment['text']}\n\n"
            f"{moment['timestamp']} into {episode.get('title', '')}\n"
            f"{moment.get('url', '')}\n")


_WINDOW = re.compile(r"(\d+:\d\d(?::\d\d)?)\s*→\s*(\d+:\d\d(?::\d\d)?)")


def _seconds(stamp: str) -> float:
    total = 0.0
    for part in stamp.split(":"):
        total = total * 60 + float(part)
    return total


def edges(episode: dict, opened: float, closed: float) -> tuple[str, str]:
    """The first and last words the clip actually contains.

    The reason this exists: the only way to know whether a clip opens
    mid-sentence used to be to watch it, and watching four clips to reject
    one is most of the work the script was written to remove. The
    transcript already knows what is in the window, so the check can be
    read in a second instead.
    """
    inside = [s for s in episode.get("segments") or []
              if opened <= float(s.get("t", 0)) < closed]
    words = " ".join(s.get("text", "") for s in inside).split()
    if not words:
        return "", ""
    return " ".join(words[:9]), " ".join(words[-7:])


def cut(episode_id: str, stamp: str, out: Path,
        seconds: float) -> tuple[float, float] | None:
    """Hand it to make_clip, which owns every decision about encoding.

    Returns the window it actually cut, parsed from make_clip's own report
    rather than recomputed here — the snapping lives in app/clipper and a
    second copy of that arithmetic would drift from it.
    """
    run = subprocess.run(
        [sys.executable, str(CLIPPER), "--episode", episode_id,
         "--at", stamp, "--seconds", str(seconds), "--out", str(out)],
        capture_output=True, text=True)
    if run.returncode != 0:
        # The whole run must not die because one moment sat too near the
        # end of the file to cut. Say which, keep going.
        tail = (run.stderr or run.stdout or "").strip().splitlines()
        print(f"      could not cut {stamp}: {tail[-1] if tail else '?'}")
        return None
    found = _WINDOW.search(run.stdout or "")
    if not found:
        return (0.0, 0.0)
    return _seconds(found.group(1)), _seconds(found.group(2))


def cut_all(moments: list[dict], episode: dict, folder: Path,
            seconds: float) -> int:
    """Cut each moment, and print enough to judge it without watching it."""
    folder.mkdir(parents=True, exist_ok=True)
    print()
    done = 0
    for i, m in enumerate(moments, 1):
        out = folder / f"{i:02d}.mp4"
        print(f"   cutting {i}/{len(moments)}  {m['timestamp']} …")
        window = cut(episode["episode_id"], m["timestamp"], out, seconds)
        if window is None:
            continue
        (folder / f"{i:02d}.txt").write_text(caption(m, episode))
        done += 1
        opens, closes = edges(episode, *window)
        if opens:
            # Printed so the clip can be judged without opening it. A
            # lowercase first word, or a last word that is not the end of
            # a sentence, is what a bad cut looks like in text.
            print(f"      opens: “{opens}…”")
            print(f"      ends:  “…{closes}”")
    print(f"\n  {done} clip(s) and captions in {folder}")
    if done:
        print(f"  open it:  open {folder}")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", help="episode id (default: newest)")
    ap.add_argument("--count", type=int, default=4,
                    help="how many moments to cut (default 4)")
    ap.add_argument("--seconds", type=float, default=50.0,
                    help="clip length before snapping to the sentence")
    ap.add_argument("--out", help="folder (default ~/Desktop/clips-<id>)")
    ap.add_argument("--dry-run", action="store_true",
                    help="pick the moments and print them; cut nothing")
    # Two-step, because "is this the important moment of the episode" is
    # not a question the picker can answer and should not pretend to. It
    # reads a sample of a four-hour show and has no idea which sixty
    # seconds anyone will care about. What it is good at is producing a
    # shortlist that is mostly not rubbish.
    #
    # So: --shortlist writes the candidates to the folder and cuts nothing,
    # and --pick cuts the ones a person chose. The shortlist is saved
    # rather than re-picked because the picker is not deterministic — ask
    # twice and you get different moments, so "run it again and cut number
    # 3" would cut a different number 3.
    ap.add_argument("--shortlist", type=int, metavar="N",
                    help="offer N candidates, save them, cut nothing")
    ap.add_argument("--pick", metavar="1,4,7",
                    help="cut these from the saved shortlist")
    args = ap.parse_args()

    if not EPISODES.exists():
        raise SystemExit(f"  {EPISODES} is missing")
    episodes = json.loads(EPISODES.read_text())
    by_id = {e["episode_id"]: e for e in episodes}
    episode = by_id.get(args.episode) if args.episode else newest(episodes)
    if episode is None:
        raise SystemExit(f"  no episode {args.episode!r} in episodes.json")

    print(f"  {episode.get('title', '')[:66]}")
    print(f"  {episode['episode_id']}  published {episode.get('published_at')}"
          f"  {len(episode.get('segments') or [])} segments\n")

    folder = Path(args.out).expanduser() if args.out else (
        Path.home() / "Desktop" / f"clips-{episode['episode_id']}")
    saved = folder / "moments.json"

    if args.pick:
        if not saved.exists():
            raise SystemExit(f"  no shortlist at {saved}. run --shortlist first.")
        pool = json.loads(saved.read_text())
        try:
            wanted = [int(x) for x in args.pick.replace(" ", "").split(",") if x]
        except ValueError:
            raise SystemExit(f"  could not read --pick {args.pick!r}; want 1,4,7")
        bad = [i for i in wanted if not 1 <= i <= len(pool)]
        if bad:
            raise SystemExit(f"  no candidate {bad} — the shortlist has "
                             f"1..{len(pool)}")
        return cut_all([pool[i - 1] for i in wanted], episode, folder,
                       args.seconds)

    settings = get_settings()
    client = AsyncAnthropic(**anthropic_client_kwargs(settings))

    # Asked three times rather than once, keeping whatever is new each
    # round. The picker is not deterministic: the same episode returned
    # three moments, then none, then one, on three consecutive runs. Once
    # is fine for a one-off. This is meant to be run every Thursday
    # morning against an episode that aired overnight, and "no moments
    # today" would be indistinguishable from a dull episode.
    wanted_count = args.shortlist or args.count
    moments: list[dict] = []
    seen: set[str] = set()
    for attempt in range(1, ATTEMPTS + 1):
        found = await highlights_for(client, settings.anthropic_model,
                                     episode, wanted_count)
        # safe_to_post is make_highlights' own filter for a moment that
        # reads badly out of context. It is the reason that pool is
        # postable, so it applies here rather than being re-litigated.
        for m in found:
            stamp = m.get("timestamp")
            if stamp and stamp not in seen and safe_to_post(m.get("text", "")):
                seen.add(stamp)
                moments.append(m)
        if len(moments) >= wanted_count:
            break
        if attempt < ATTEMPTS:
            print(f"   have {len(moments)}/{wanted_count} after {attempt} "
                  f"pass(es); asking again…")
    moments = moments[:wanted_count]
    if not moments:
        print(f"  no moment cleared the bar in {ATTEMPTS} passes. "
              f"nothing cut.")
        return 0

    for i, m in enumerate(moments, 1):
        print(f"   {i}. [{m['timestamp']}] {m['text'][:96]}")

    if args.dry_run:
        print("\n  dry run. drop --dry-run to cut these.")
        return 0

    if args.shortlist:
        folder.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(moments, indent=1))
        print(f"\n  {len(moments)} candidates saved to {saved}")
        print(f"  cut the ones worth cutting:\n"
              f"    scripts/clip_episode.py --episode {episode['episode_id']}"
              f" --pick 1,3")
        return 0

    return cut_all(moments, episode, folder, args.seconds)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
