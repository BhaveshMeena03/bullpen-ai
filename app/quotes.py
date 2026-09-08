"""Did he actually say that?

A quote goes around with a name on it. Sometimes the name is wrong, and
sometimes the words are. There is no way to settle it by argument, and the
only thing that settles it is the recording — which is what this archive
is. So: paste the quote, get back where it was said, or get back that it
was not.

This is deliberately NOT the semantic search. Retrieval answers "what was
said about X" and is meant to be forgiving; a paraphrase should still find
the passage. Verification is the opposite job. "Close enough" is exactly
the failure being checked for, because a misquote that survives is one
somebody rephrased until it said what they wanted. So this matches on the
words themselves, in order, and reports how much of the claim it actually
found rather than a similarity score nobody can interpret.

What it can say:

  found       most of the claim appears verbatim, in one place. Comes back
              with the episode, the second, and the surrounding words as
              the transcript has them, so a reader can check the checker.
  partial     a real but incomplete match. The interesting case: usually a
              genuine line with an ending bolted on, and printing the true
              words beside the claim shows exactly where it stops being a
              quote.
  not_found   no run of it is in the archive. NOT the same as "he never
              said it" — this archive is 38 broadcasts and 11 interviews,
              not everything he has ever said — and the reply says so,
              because an archive that overstates its own coverage is worse
              than no archive.

Matching is over normalised text: lowercase, punctuation dropped, runs of
space collapsed. That is the same normalisation dedupe.py uses, and it is
what makes "$100 million" and "100 million" the same claim, which they are
when spoken.
"""

from __future__ import annotations

import bisect
import re

# Same shape as dedupe.py's. Punctuation is transcription's guess, not
# something the speaker said, so it must not decide whether a quote matches.
_CLEAN = re.compile(r"[^a-z0-9 ]")
_SPACE = re.compile(r"\s+")

# Below this, a "quote" is a phrase that will collide by chance. Six words
# of ordinary speech appear in 67 hours of talking for no reason at all,
# and reporting that as a verified quote would make the tool a liar in the
# one direction it cannot afford.
MIN_WORDS = 7

# Share of the claim that must appear, in one contiguous run, to call it
# found. Not 1.0: a real quote is nearly always topped and tailed by a few
# words the quoter dropped, and demanding the whole string would fail the
# true quotes far more often than it would catch the false ones.
FOUND = 0.80
# Deliberately low. The case this catches is a real line with an invented
# ending bolted on — measured at 0.44 on "best case five years, worst case
# ten years, and I guarantee we will be there by 2027", against 0.28 for a
# quote invented outright. Calling the first one "not_found" hides the most
# useful thing the tool can say: most of this is real, and here is the word
# where it stops being real.
PARTIAL = 0.35


def normalise(text: str) -> str:
    return _SPACE.sub(" ", _CLEAN.sub(" ", (text or "").lower())).strip()


class Haystack:
    """One episode, flattened once, with a way back to the timestamps.

    The flattened string is what gets searched, because Python's substring
    search is C and a per-word loop in Python is not. `_starts` records the
    character offset at which each segment begins, so a hit anywhere in
    that string can be turned back into a segment — and therefore a second
    of audio — with a bisect instead of a scan.
    """

    __slots__ = ("episode", "text", "_starts", "_index")

    def __init__(self, episode: dict) -> None:
        self.episode = episode
        parts, starts, index, at = [], [], [], 0
        for i, seg in enumerate(episode.get("segments") or []):
            words = normalise(seg.get("text", ""))
            if not words:
                continue
            starts.append(at)
            index.append(i)
            parts.append(words)
            at += len(words) + 1
        # Padded, and every search is for " phrase " rather than "phrase".
        # Without the boundaries a substring search matches inside a word —
        # the claim word "on" is present in "python" — which quietly grows
        # a run past where the real words stop. Offsets are corrected for
        # the pad in segment_at.
        self.text = " " + " ".join(parts) + " "
        self._starts = starts
        self._index = index

    def segment_at(self, offset: int) -> int | None:
        """Which segment covers this character offset."""
        if not self._starts:
            return None
        # -1 undoes the leading pad added in __init__.
        pos = bisect.bisect_right(self._starts, max(offset - 1, 0)) - 1
        return self._index[max(pos, 0)]


def _longest_run(claim: list[str], hay: str) -> tuple[int, int, int]:
    """(words matched, offset in hay, start word in claim) for the best run.

    Every start position is tried, and each is grown by binary search
    rather than a word at a time: the run either is or is not present at a
    given length, so the length is a sorted predicate and costs log(n)
    substring searches instead of n.
    """
    best = (0, -1, 0)
    n = len(claim)
    for i in range(n):
        # Cannot beat what we already have from here.
        if n - i <= best[0]:
            break
        lo, hi, found_at = 1, n - i, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            at = hay.find(" " + " ".join(claim[i:i + mid]) + " ")
            if at >= 0:
                found_at, lo = at + 1, mid + 1
            else:
                hi = mid - 1
        run = hi
        if run > best[0]:
            best = (run, found_at, i)
    return best


# A quote is almost never one unbroken run, and the reasons are ordinary:
# the transcript carries an interjection the quoter left out ("...on Mars.
# Hmm. Best case is about five years"), or the quoter elided a clause with
# no ellipsis. Requiring one contiguous run scored both of those the same
# as an invention, which is the wrong error for a verifier to make — a
# false "not_found" on a real quote is how it loses the argument it exists
# to settle. So runs are summed, subject to two limits.
MIN_RUN = 4          # shorter than this is coincidence, not quotation
WINDOW = 3000        # chars: runs must be from the same stretch of talk

# On EVERY answer, including the ones that found nothing — that is the one
# most likely to be quoted back as "the archive says he never said it".
COVERAGE = ("checked against this archive only; absence here is not proof "
            "the words were never said")


def _cover(claim: list[str], hay: str) -> int:
    """Total claim words found in `hay` as runs of at least MIN_RUN.

    Greedy longest-first, then the same on what is left either side. The
    pieces cannot overlap because each recursion gets a disjoint slice of
    the claim.
    """
    if len(claim) < MIN_RUN:
        return 0
    run, _, start = _longest_run(claim, hay)
    if run < MIN_RUN:
        return 0
    return (run + _cover(claim[:start], hay)
            + _cover(claim[start + run:], hay))


def check(claim: str, episodes: list[dict],
          speaker_map: dict | None = None) -> dict:
    """Where this was said, or that it was not."""
    words = normalise(claim).split()
    if len(words) < MIN_WORDS:
        return {"verdict": "too_short",
                "detail": f"give at least {MIN_WORDS} words — anything "
                          f"shorter matches by chance",
                "claim_words": len(words), "coverage": COVERAGE}

    best = None
    for episode in episodes:
        hay = Haystack(episode)
        if not hay.text:
            continue
        # Anchor on the longest run, then score only the stretch of talk
        # around it. Scoring the whole episode would let a common phrase
        # from an unrelated hour top up a quote that was never said.
        run, offset, start = _longest_run(words, hay.text)
        if run < MIN_RUN:
            continue
        near = hay.text[max(0, offset - WINDOW):offset + WINDOW]
        covered = max(run, _cover(words, near))
        if not best or covered > best[0]:
            best = (covered, offset, start, run, hay)

    if not best or best[0] == 0:
        return {"verdict": "not_found", "matched_words": 0,
                "matched_share": 0.0, "claim_words": len(words),
                "detail": "no run of these words is in the archive",
                "coverage": COVERAGE}

    covered, offset, start, run, hay = best
    share = covered / len(words)
    episode, seg_i = hay.episode, hay.segment_at(offset)
    segments = episode.get("segments") or []
    seconds = float(segments[seg_i].get("t", 0)) if seg_i is not None else 0.0

    speaker = None
    if speaker_map and seg_i is not None:
        speaker = (speaker_map.get(episode["episode_id"]) or {}).get(str(seg_i))

    # The transcript's own words around the hit, so the answer can be
    # checked rather than trusted. Slightly wider than the match: a quote
    # that ends where the sentence does not is the thing worth seeing.
    verbatim = " ".join(s.get("text", "")
                        for s in segments[max(0, (seg_i or 0) - 1):
                                          (seg_i or 0) + 4]).strip()

    verdict = ("found" if share >= FOUND
               else "partial" if share >= PARTIAL else "not_found")
    return {
        "verdict": verdict,
        "matched_words": covered,
        "longest_run": run,
        "claim_words": len(words),
        "matched_share": round(share, 3),
        "matched_text": " ".join(words[start:start + run]),
        "episode_id": episode["episode_id"],
        "title": episode.get("title"),
        "url": episode.get("url"),
        "seconds": seconds,
        "speaker": speaker,
        "verbatim": verbatim,
        "coverage": COVERAGE,
    }
