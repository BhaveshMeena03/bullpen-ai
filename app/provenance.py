"""Two guards on what is allowed to become a citation.

The product's whole claim is "he said this, at this second". A citation is
only worth anything if the words were said and the person said them, so
these check the two ways that goes wrong in practice. Both were found by
running the pipeline over real material rather than reasoned about in
advance.

  hallucinated speech   Whisper writes words over silence. On a Tesla
                        earnings call it produced "Let's get started."
                        twenty-nine times across eight and a half minutes
                        of hold music -- nine minutes of speech nobody
                        said, timestamped as confidently as the rest.

  the wrong recording   A search for one Lex Fridman episode returned the
                        original and a channel called "Game Time" hosting
                        an identical 1,965-second copy. Ingest the copy and
                        the archive holds a re-upload it cannot tell from
                        the source, cited just as precisely.

Neither is hypothetical and neither announces itself downstream: a
fabricated line and a real one are the same shape by the time they reach
retrieval.
"""

from __future__ import annotations

import collections
import re

# How many times a line may repeat before it is read as Whisper filling
# silence rather than as speech.
#
# Real conversation repeats short words constantly -- "Yeah." nine times
# in a 33-minute interview, "Yes." four, "Thank you." four -- so a low
# threshold would delete genuine speech. The hold music produced 29
# identical copies of a full sentence, which is a different animal. The
# line sits above what conversation does and far below what silence does.
REPEAT_LIMIT = 12
# Interjections of two words or fewer are exempt at any count. "Yeah"
# fifteen times across an hour is a person agreeing, not a hallucination,
# and the same goes for "Yes." and "Thank you." -- all three measured on a
# real interview.
#
# Two, not three. The line this guard was built for is "Let's get
# started.", which is exactly three words, so exempting three-word phrases
# exempted the only case it had ever seen. Caught by the test carrying the
# real example rather than a made-up one.
INTERJECTION_WORDS = 2


def drop_hallucinated(segments: list[dict]) -> tuple[list[dict], list[str]]:
    """(kept, what was dropped) — segments Whisper invented over silence.

    Identified by exact repetition of a full phrase, because that is what
    the failure looks like: the decoder latches onto one output and emits
    it for every window of quiet. A speaker repeating themselves varies
    the wording; a hallucination does not vary at all.
    """
    counts = collections.Counter(s.get("text", "").strip() for s in segments)
    banned = {
        text for text, n in counts.items()
        if n >= REPEAT_LIMIT
        and len(text.split()) > INTERJECTION_WORDS
    }
    kept = [s for s in segments if s.get("text", "").strip() not in banned]
    notes = [f"{text!r} x{counts[text]}" for text in sorted(banned)]
    return kept, notes


# Channels that are the original publisher of what they post. A recording
# is only ingested from the account that made it.
#
# Not a quality judgement about anyone else. A re-upload is usually the
# same audio, and that is the problem: it is indistinguishable downstream
# from the source, so a citation into it claims a provenance the archive
# cannot actually stand behind. Verifying one costs more than transcribing
# the original again.
PUBLISHERS = frozenset({
    "Tesla", "SpaceX", "Lex Fridman", "PowerfulJRE", "TED",
    "All-In Podcast", "The Economist", "VideoFromSpace",
    "Nikhil Kamath", "Dwarkesh Patel",
})


def is_original_publisher(channel: str) -> bool:
    """Whether this channel is the one that made the recording."""
    return (channel or "").strip() in PUBLISHERS


_RE_UPLOAD_MARKS = re.compile(
    r"""(?ix)\b(?: full\s+episode\s+reupload | re-?upload | mirror
                 | compilation | best\s+of | motivational
                 | reaction | debunk(?:ing|ed)? | explained\s+by
    )\b""")


def looks_like_someone_elses_cut(title: str) -> bool:
    """A title that advertises itself as not being the original.

    A second line of defence behind the channel list, for the case the
    list has not seen a publisher yet. Deliberately narrow: it catches
    what a re-uploader writes on purpose, not everything unusual.
    """
    return bool(_RE_UPLOAD_MARKS.search(title or ""))


def admissible(channel: str, title: str) -> tuple[bool, str]:
    """(allowed, why not). The single call an ingest should make."""
    if looks_like_someone_elses_cut(title):
        return False, "the title says it is someone's cut, not the recording"
    if not is_original_publisher(channel):
        return False, f"{channel!r} is not the channel that made this"
    return True, ""
