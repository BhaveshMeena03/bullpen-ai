"""Which moment of the archive is this clip?

A clipper posts thirty seconds of the show and writes "must watch". Asked
what they are talking about, the bot had nothing to read: the caption is
three words, so it fell through to the newest episode and answered with
3,890 characters about it. That is right exactly when the clip happens to
come from the newest show, and confidently wrong otherwise.

The clip itself is the missing half of the question. Not by understanding
the video -- there is nothing to understand -- but by IDENTIFYING it. If
the clip came from a show in the archive, the archive already holds the
transcript around it, including the part the clipper cut off. That is the
whole value: answering with what was said after the clip ended.

Matching whole lines does not work, for the reason align_recordings.py
records: the clip is transcribed separately from the archive copy, so
Whisper splits and spells it differently. Measured on a real clipper's
post, Whisper dropped a phrase the archive kept and merged two others --
and 138 eight-word runs still matched, all inside one 69-second span of
one episode. Runs survive what lines do not.

Deliberately NOT align_recordings.offset(). That function needs
MIN_MATCHES = 200 because it aligns two copies of a four-hour show, where
a real pair yields tens of thousands. A 71-second clip yielded 231 runs
in total. Reusing that threshold would refuse every clip ever posted.

The rule here is different and the safety comes from a different place:
a clip from an indexed show lands many unique runs in ONE tight span,
while a clip from a show nobody indexed lands almost none and what it
does land scatters. So require both -- a floor on matches, and agreement
that they describe a single stretch of tape. Anything else returns None,
and None must mean "say you cannot place it" rather than "guess".
"""

from __future__ import annotations

import collections
import re

GRAM = 8

# A clip has to clear this to be placed at all. Kaiz's 71-second clip
# gave 138; a clip from an unindexed show gives a handful of accidental
# matches on common phrasing. Twelve sits far above the accident rate and
# far below anything real, and it is a floor rather than a target: more
# is better, fewer is a refusal.
MIN_UNIQUE_RUNS = 12

# Matched runs must describe one stretch of tape, not scattered hits
# across an episode. A real clip's runs land in a span about as long as
# the clip; coincidental matches spread over hours. Judged as a fraction
# inside the tightest window rather than an absolute, because a
# five-minute clip legitimately spans five minutes.
MIN_IN_SPAN = 0.80

# How far either side of the densest point still counts as the same
# stretch. Generous: the two transcriptions disagree about where a
# sentence starts, and align_recordings.py measured that disagreement at
# under fifteen seconds on a known-good pair.
SPAN_SLACK_SECONDS = 20.0


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def runs(text: str) -> list[str]:
    """Every eight-word run in order. Order is kept, not a set: a run
    repeated inside the clip is still one piece of evidence per
    occurrence, and dropping duplicates here would hide that."""
    got = words(text)
    return [" ".join(got[i:i + GRAM]) for i in range(len(got) - GRAM + 1)]


def _index(episode: dict,
           wanted: set[str] | None = None) -> dict[str, list[float]]:
    """run -> the seconds it starts at, for one episode.

    `wanted` narrows the work to the runs the caller will actually read.
    Indexing the whole archive unnarrowed builds 1.28 million distinct
    runs and peaks at 260 MB, and the bot shares a process with the web
    app, so a reply that placed a clip would briefly cost the site a
    quarter of a gigabyte. Narrowed to one clip's runs it is 6 MB.

    This cannot change a verdict. The filter decides which runs get
    RECORDED, never how many occurrences of a recorded run get counted --
    so the uniqueness rule below, which only trusts a run appearing
    exactly once in the episode, still sees every occurrence of the runs
    it judges. Checked against the unnarrowed index over eight clips
    against all 39 episodes: identical dicts, and identical place()
    verdicts including every refusal.
    """
    stamped: list[tuple[str, float]] = []
    for segment in episode.get("segments") or ():
        at = float(segment.get("t", 0.0))
        for word in words(segment.get("text")):
            stamped.append((word, at))
    found: dict[str, list[float]] = collections.defaultdict(list)
    # A run can only be wanted if its first word starts one, and that test
    # is a set lookup against ~200 words rather than joining eight of them.
    # Building the key is the expensive part; this skips it on the ~99% of
    # positions that cannot match.
    firsts = None if wanted is None else {r.split(" ", 1)[0] for r in wanted}
    for i in range(max(0, len(stamped) - GRAM + 1)):
        if firsts is not None and stamped[i][0] not in firsts:
            continue
        key = " ".join(w for w, _ in stamped[i:i + GRAM])
        if wanted is not None and key not in wanted:
            continue
        found[key].append(stamped[i][1])
    return found


def _densest(times: list[float],
             window_seconds: float) -> tuple[float, float, float]:
    """(share inside the best window, start, end).

    The window is a FIXED duration passed in by the caller -- the clip's
    own length plus slack. The first version computed it from the matches
    being judged:

        hi = anchor + (max(order) - min(order)) + SPAN_SLACK_SECONDS

    which is the spread of those very matches, so the window always
    covered all of them and the share was 1.00 by construction. The gate
    rejected nothing, and setting MIN_IN_SPAN to 0.0 changed no result
    because the value was never below it. Caught by a fixture stitched
    from three points four hours apart, which scored a perfect 1.00 and
    was placed as though it were one continuous clip.

    A window judging the data cannot be derived from the data.
    """
    if not times:
        return 0.0, 0.0, 0.0
    order = sorted(times)
    best = (0, order[0], order[0])
    for anchor in order:
        hi = anchor + window_seconds
        inside = [t for t in order if anchor <= t <= hi]
        if len(inside) > best[0]:
            best = (len(inside), inside[0], inside[-1])
    return best[0] / len(order), best[1], best[2]


def place(transcript: str, episodes: list[dict],
          duration_seconds: float | None = None) -> dict | None:
    """Where this clip came from, or None when it cannot be placed.

    None is the important return. A clip from a show that was never
    indexed matches almost nothing, and answering it from the newest
    episode is the bug this module exists to remove.

    Only runs that are unique IN THE EPISODE are counted, the same rule
    align_recordings.offset() uses: a phrase somebody repeats through the
    night would otherwise pair the clip's single instance with whichever
    of the three it met first and invent a location.
    """
    wanted = runs(transcript)
    if len(wanted) < MIN_UNIQUE_RUNS:
        return None
    asked = set(wanted)

    # How long a stretch of tape this clip is allowed to occupy. The
    # caller knows it -- post_by_id returns duration_ms -- and passing the
    # real figure is what makes the span gate mean anything. Without it,
    # estimate from the transcript: conversational speech runs about 2.5
    # words a second, measured across this archive. Either way the window
    # is decided BEFORE looking at the matches, which is the property the
    # first version lost.
    if duration_seconds is None:
        duration_seconds = len(words(transcript)) / 2.5
    window = duration_seconds + SPAN_SLACK_SECONDS

    scored: list[dict] = []
    for episode in episodes:
        index = _index(episode, asked)
        times = [index[run][0] for run in asked
                 if len(index.get(run, ())) == 1]
        if len(times) < MIN_UNIQUE_RUNS:
            continue
        share, start, end = _densest(times, window)
        if share < MIN_IN_SPAN:
            continue
        scored.append({
            "episode_id": episode.get("episode_id", ""),
            "title": episode.get("title", ""),
            "matches": len(times),
            "start": start,
            "end": end,
            "share_in_span": share,
        })
    if not scored:
        return None
    scored.sort(key=lambda row: row["matches"], reverse=True)
    best = scored[0]
    # Two episodes both matching well means the archive holds the same
    # show twice -- a broadcast and its YouTube cut, which dedupe.py
    # already knows about -- or the match is not trustworthy. Either way
    # the caller gets the strongest one and how far ahead it was.
    best["runner_up"] = scored[1]["matches"] if len(scored) > 1 else 0
    return best
