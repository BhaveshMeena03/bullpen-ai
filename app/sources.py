"""Never name a recording the search did not return.

The Musk archive fails in a way the podcast archive cannot. The model
has heard these interviews before -- they are famous, and they are in
its training data -- so when it writes an answer it can name a
plausible one from memory. Measured over twenty-five questions, six of
the eight recall answers cited at least one recording that was never
retrieved, always in ADDITION to the ones that were:

    shown only 2024 Lex Fridman #438, an answer about working hours
    cited "the 2021 Joe Rogan episode"

    shown 2018 Joe Rogan, 2019 Lex Fridman and 2021 Joe Rogan, an
    answer about Mars cited "the 2021 Lex Fridman episode", which
    pairs a year it was given with a show it was given and names a
    recording that does not exist here

Market Bubble never does this, and cannot: the model has no idea what
was said on episode 13, so it has nothing to invent with. This is the
cost of indexing something the model already knows.

The quotes in those answers hold. `quotes_hold` places them near the
cited second in a recording that DID come back, so the material is
real and only the label on it is wrong. That is what makes this
fixable without touching the substance.

Demoted, not corrected, for the reason attribution.correct demotes: we
know the claim is wrong, and we do not know which recording the writer
meant. Naming a different one would be a second guess, and a confident
wrong source is worse than a vague true one. "the 2018 Joe Rogan
episode" becomes "one of these conversations" and the timestamp, quote
and link are left exactly as they were.
"""

from __future__ import annotations

import re

# The only two shows in this archive. Hardcoded rather than read off
# the returned labels, because the whole failure is the model naming a
# show that did NOT come back, and a pattern built from what came back
# could not see it.
_SHOW = r"joe\s+rogan|rogan|lex\s+fridman|lex"
_YEAR = r"20[0-2]\d"
_NOUN = r"episode|conversation|podcast|interview|appearance"

# "the 2018 Joe Rogan episode"
_YEAR_SHOW = re.compile(
    rf"(?i)\b(?:the\s+)?({_YEAR})\s+({_SHOW})(?:\s+(?:{_NOUN}))?\b")
# "the Joe Rogan episode from 2018"
_SHOW_YEAR = re.compile(
    rf"(?i)\b(?:the\s+)?({_SHOW})(?:\s+(?:{_NOUN}))?\s+(?:from|in)\s+({_YEAR})\b")
# "the 2020 episode" -- a recording named by date alone.
_YEAR_ONLY = re.compile(rf"(?i)\b(?:the\s+)?({_YEAR})\s+(?:{_NOUN})\b")

_DEMOTED = "one of these conversations"

# "in that same 2021 conversation" is a back-reference, not a fresh
# claim, and swapping the whole phrase for the standard wording gave
# "that same one of these conversations". After "that same" the
# recording has already been established, so only the label goes.
_BACK_REFERENCE = re.compile(r"(?i)\b(?:that|the)\s+same\s+$")
_DEMOTED_AGAIN = "conversation"


def _key(show: str) -> str:
    return "rogan" if "rogan" in show.lower() else "lex"


def _given(hits) -> list[str]:
    """The source label of every recording the search returned."""
    out = []
    for hit in hits or ():
        label = getattr(hit, "source", None)
        if not label:
            title = getattr(hit, "title", "") or ""
            aired = getattr(hit, "published_at", "") or ""
            label = f"{aired[:4]} {title}"
        out.append(label.lower())
    return out


def correct(answer: str, hits) -> tuple[str, list[str]]:
    """Demote any recording the passages do not contain.

    Returns the answer and what changed, for the log. An answer naming
    only recordings that came back is returned identical, by identity.
    """
    if not answer or not hits:
        return answer, []
    labels = _given(hits)
    if not labels:
        return answer, []

    changes: list[str] = []

    def sweep(pattern: re.Pattern, holds) -> None:
        nonlocal answer
        # Reverse, so replacing one phrase does not shift the offsets of
        # the ones still to check.
        for found in reversed(list(pattern.finditer(answer))):
            if holds(found):
                continue
            word = (_DEMOTED_AGAIN
                    if _BACK_REFERENCE.search(answer[:found.start()])
                    else _DEMOTED)
            changes.append(f"{found.group(0)!r} -> {word!r}")
            answer = answer[:found.start()] + word + answer[found.end():]

    # Pairs first: they are the specific claim, and demoting the pair
    # leaves no bare year behind for the looser sweep to act on twice.
    sweep(_YEAR_SHOW, lambda m: any(
        m.group(1) in l and _key(m.group(2)) in l for l in labels))
    sweep(_SHOW_YEAR, lambda m: any(
        m.group(2) in l and _key(m.group(1)) in l for l in labels))
    sweep(_YEAR_ONLY, lambda m: any(m.group(1) in l for l in labels))

    return answer, list(reversed(changes))
