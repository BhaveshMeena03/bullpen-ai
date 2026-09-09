"""The timestamped contents of an episode, ready to paste into a post.

    .venv/bin/python scripts/make_chapters.py                  # newest
    .venv/bin/python scripts/make_chapters.py --episode UIcjWPG7eJ8
    .venv/bin/python scripts/make_chapters.py --count 30 --offset 1:12

Market Bubble posts these by hand:

    0:20 Who is @jboogx_creative?
    4:08 How and why jboog became an AI creator
    7:22 Making money with AI

X turns a bare timestamp in the text of a post into a link that seeks the
video ATTACHED TO THAT POST. No url is involved, which is the whole reason
it works where a citation does not: a url to another post is rendered as
a quote card, and a card cannot carry a timestamp.

So the timestamps here are only meaningful against the video in the post
they are pasted into. Attach the full episode and they map straight
across. Attach a clip and they do not -- pass --offset with the second the
clip starts and every line is rebased, because a chapter list that is
three minutes out is worse than none.

Somebody scrubbing a four-hour stream by hand gives up around twenty
entries, which is why the hand-made ones stop early. This reads the
transcript, so it costs a minute and reaches the end.

Prints. Nothing is posted, and nothing is written to the archive.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from anthropic import AsyncAnthropic  # noqa: E402

from app.config import anthropic_client_kwargs, get_settings  # noqa: E402
from app.podcast import _timestamp  # noqa: E402
from clip_episode import newest  # noqa: E402

EPISODES = [ROOT / "data" / "episodes.json",
            ROOT / "data" / "elon_episodes.json"]

# Lines sent to the model. Same budget make_highlights uses, and spread
# across the whole episode by stride rather than truncated -- truncating
# is how that one ended up never reading the last ninety minutes of a
# four-hour show.
SAMPLE_LINES = 1400

# Chapters closer together than this are the same topic described twice.
# Market Bubble's own lists run about one every two to four minutes.
MIN_GAP_SECONDS = 60

PROMPT = """\
Below is a transcript of one episode, sampled evenly across its whole \
length. Every line begins with its timestamp in square brackets.

Produce a table of contents: about {n} entries, in order, marking where \
the conversation MOVES TO A NEW TOPIC. This is a chapter list for someone \
deciding which part to watch.

Rules:
1. Use ONLY timestamps that appear in the transcript below, copied exactly \
as written, including the brackets. Do not compute or adjust a timestamp. \
If a topic starts between two lines, use the line that starts it.
2. One entry per line, formatted exactly:  [0:14:22] Title of the section
3. Titles are 3 to 8 words, plain and descriptive. Say what is discussed, \
not why it is interesting. "Rotating into gold and US oil", not "You won't \
believe this gold take".
4. Do not invent a topic that is not discussed. Do not editorialise.
5. Space them out. Two entries a minute apart are one topic, not two.
6. Cover the whole episode, including the end.
Output the list and nothing else."""


def episodes() -> list[dict]:
    out: list[dict] = []
    for source in EPISODES:
        if source.exists():
            out += json.loads(source.read_text())
    return out


def sample(episode: dict) -> tuple[str, dict[str, float]]:
    """(the transcript to send, timestamp -> seconds for what was sent).

    The second value is the whole validation strategy. The model is told
    to copy a timestamp it was shown, so anything it returns that is not
    in this map was invented, and can be dropped without judgement.
    """
    lines, stamps = [], {}
    segments = [s for s in episode.get("segments") or []
                if (s.get("text") or "").strip()]
    stride = max(1, -(-len(segments) // SAMPLE_LINES))
    for seg in segments[::stride][:SAMPLE_LINES]:
        stamp = _timestamp(seg["t"])
        stamps[stamp] = float(seg["t"])
        lines.append(f"[{stamp}] {seg['text'].strip()}")
    return "\n".join(lines), stamps


_ENTRY = re.compile(r"^\s*\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s+(.{3,120}?)\s*$")


def parse(text: str, stamps: dict[str, float],
          offset: float) -> list[tuple[float, str]]:
    """Model output -> [(seconds, title)], invented timestamps dropped."""
    out: list[tuple[float, str]] = []
    for line in (text or "").splitlines():
        found = _ENTRY.match(line)
        if not found:
            continue
        stamp, title = found.group(1), found.group(2).strip()
        if stamp not in stamps:
            # It was asked to copy one it was shown. Anything else is a
            # timestamp it made up, and a chapter list that points at the
            # wrong minute is worse than a shorter one.
            print(f"     dropped (not in the transcript): [{stamp}] {title[:48]}")
            continue
        out.append((stamps[stamp] - offset, title))

    out.sort(key=lambda x: x[0])
    spaced: list[tuple[float, str]] = []
    for seconds, title in out:
        if seconds < 0:
            continue                       # before the clip starts
        if spaced and seconds - spaced[-1][0] < MIN_GAP_SECONDS:
            continue
        spaced.append((seconds, title))
    return spaced


# Words that say nothing about what a section is about, so their presence
# or absence in the transcript proves nothing either way.
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "his", "her", "its", "their", "this", "that", "these", "those", "is",
    "are", "was", "were", "be", "been", "as", "at", "by", "from", "about",
    "into", "over", "versus", "vs", "how", "why", "what", "when", "who",
    "discussion", "discussed", "discussing", "talks", "talking", "segment",
    "intro", "introduces", "introduction", "thoughts", "review", "check",
    "recap", "wrap", "wrapping", "opens", "opening", "closing", "more",
}

# How much of a title's substance has to actually appear near its
# timestamp. Below this the title is describing something else -- the
# failure found by hand on a first run, where "Venice AI and investing in
# AI platforms" sat on a passage about Spotify and managing a creator.
TITLE_SUPPORT = 0.34


def support(title: str, episode: dict, seconds: float,
            before: float = 30.0, after: float = 150.0) -> tuple[float, list[str]]:
    """(share of the title's content words found near it, the missing ones).

    Deliberately crude. It is not judging whether the title is a GOOD
    description, only whether the words it uses are spoken anywhere near
    where it points -- which is enough to catch a title attached to the
    wrong minute, and cheap enough to run on every line.
    """
    words = [w for w in re.findall(r"[a-z0-9']+", title.lower())
             if w not in _STOP and len(w) > 2]
    if not words:
        return 1.0, []
    window = " ".join(
        s.get("text", "") for s in episode.get("segments") or []
        if seconds - before <= float(s.get("t", 0)) <= seconds + after
    ).lower()
    missing = [w for w in words if w[:5] not in window]
    return 1 - len(missing) / len(words), missing


def parse_offset(value: str | None) -> float:
    if not value:
        return 0.0
    total = 0.0
    for part in str(value).split(":"):
        total = total * 60 + float(part)
    return total


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", help="episode id (default: newest)")
    ap.add_argument("--count", type=int, default=25,
                    help="roughly how many chapters (default 25)")
    ap.add_argument("--offset", metavar="M:SS",
                    help="where the posted video starts in the recording, "
                         "when posting a clip rather than the full episode")
    ap.add_argument("--out", help="write the list to a file as well")
    args = ap.parse_args()

    all_eps = episodes()
    by_id = {e["episode_id"]: e for e in all_eps}
    episode = by_id.get(args.episode) if args.episode else newest(all_eps)
    if episode is None:
        raise SystemExit(f"  no episode {args.episode!r}")

    body, stamps = sample(episode)
    offset = parse_offset(args.offset)
    print(f"  {episode.get('title', '')[:66]}")
    print(f"  {episode['episode_id']}  {len(stamps)} lines sampled to "
          f"{_timestamp(max(stamps.values()))}"
          + (f"  offset -{_timestamp(offset)}" if offset else "") + "\n")

    settings = get_settings()
    client = AsyncAnthropic(**anthropic_client_kwargs(settings))
    response = await client.messages.create(
        model=settings.anthropic_model, max_tokens=4000,
        # Extraction, not reasoning -- and thinking is billed against
        # max_tokens, so leaving it on can spend the whole budget and
        # return nothing, which reads exactly like "no chapters here".
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content":
                   f"{PROMPT.format(n=args.count)}\n\n"
                   f"<transcript>\n{body}\n</transcript>"}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    chapters = parse(raw, stamps, offset)
    if not chapters:
        print("  no usable chapters came back.")
        return 1

    # Every line checked against the transcript at the second it points
    # at. The timestamps cannot be invented — the model had to copy one it
    # was shown — but the TITLES are its reading of a sampled region, and
    # a title on the wrong minute is the failure that makes a list worse
    # than none. Weak ones are printed with what is missing so they can be
    # fixed or dropped by reading, not by scrubbing the episode.
    weak: list[str] = []
    lines, keep = [], []
    for seconds, title in chapters:
        stamp = _timestamp(seconds + offset)
        score, missing = support(title, episode, seconds + offset)
        ok = score >= TITLE_SUPPORT
        if ok:
            keep.append((seconds, title))
        else:
            weak.append(f"{stamp} {title}  (missing: {', '.join(missing[:4])})")
        lines.append(f"{'  ' if ok else ' ?'}{_timestamp(seconds)} {title}")
    print("\n".join(lines))

    # The file is the thing that gets sent to somebody, so it holds only
    # the lines that checked out. The flagged ones stay on screen, where
    # they can be fixed by hand and pasted back in if they were right.
    listing = "\n".join(f"{_timestamp(s)} {t}" for s, t in keep)
    print(f"\n  {len(chapters)} chapters. paste into a post with the video "
          f"attached — X links the timestamps to it.")
    if weak:
        print(f"\n  {len(weak)} title(s) not supported by the transcript "
              f"where they point — check or drop these:")
        for line in weak:
            print(f"     {line}")

    # A long stretch with no entry is either one genuine segment or the
    # picker thinning out. Either way it is the thing to look at before
    # handing the list to anybody.
    gaps = [(a[0], b[0]) for a, b in zip(chapters, chapters[1:])
            if b[0] - a[0] > 900]
    for start, end in gaps:
        print(f"\n  gap: nothing between {_timestamp(start)} and "
              f"{_timestamp(end)} ({int((end - start) / 60)} minutes)")
    if args.out:
        Path(args.out).expanduser().write_text(listing + "\n")
        print(f"\n  wrote {len(keep)} verified chapter(s) to {args.out}"
              + (f" ({len(weak)} flagged one(s) left out)" if weak else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
