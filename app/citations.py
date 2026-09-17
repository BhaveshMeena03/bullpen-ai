"""Move a cited second onto the line the quote actually came from.

SYSTEM_PROMPT already forbids the failure this fixes. Rule 2 says to
cite the timestamp of the line used, "NOT the `at` attribute on the
excerpt -- that is only where the passage begins". The per-line
timestamps are in front of the model, in text_ts, put there for exactly
this. It still gets it wrong sometimes, and no amount of further
instruction is a guarantee, so this is the deterministic half.

Measured on the live index:

    "what price did they say bitcoin would hit" cited 38:36, where the
    transcript is "we had, like, the Memory, Micron, like, Sandisk" --
    the Bitcoin discussion is at 54:48

    a reply under a clip cited 1:15:21, where the show is talking about
    pump.fun resistance levels; the quoted material is at 1:18:49

Both are minutes away on a different subject, so this is not the window
being a few seconds early. The model attaches a claim to a timestamp
belonging to another passage entirely.

What this does NOT do is invent a citation, reorder an answer, or touch
a timestamp it cannot place. It only moves one that a quote contradicts,
and only when the quote matches a line well enough to prove it. Every
other answer comes back identical, by identity.
"""

from __future__ import annotations

import re

# A line inside an excerpt, with the second it starts at:
#   "[1:18:49] Ansem: i think the meme coins pair with stocks..."
_STAMPED = re.compile(r"^\[(\d[\d:]*)\]\s*(?:[A-Z][A-Za-z ]{2,20}:\s*)?(.+)$",
                      re.M)

_QUOTED = re.compile(r'"([^"\n]{18,200})"')

# Text between two quotation marks is not always a quotation. An answer
# that quotes twice in one sentence leaves prose between them, and the
# regex pairs the closing mark of the first with the opening mark of the
# second. Measured on the live index, that produced:
#
#     " in the next year, and around 29:04 he compared..."
#
# which is the answer's own narration. It anchors to a line like any
# other string and would move a citation it has no claim over. Two marks
# of a quotation that this prose cannot have: a quotation is transcript,
# so it carries no timestamp and no episode billing.
_NOT_A_QUOTE = re.compile(r"\b\d{1,2}:\d{2}\b|\|\s*Market Bubble|#\d")

# A timestamp as an answer writes one: 5:04, 1:18:49.
_CITED = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")

# Shared vocabulary needed before a quote is treated as coming from a
# line. Same threshold attribution uses, for the same reason: below this
# a quote merely resembles a line and acting on it would be a guess.
_MIN_OVERLAP = 4

# How far a citation may sit from its line before it is moved. A window
# is minutes of speech and a quote can legitimately be a little after
# the second the answer names, so this only fires on a real miss. The
# measured failures were 3 and 16 minutes out.
_SLACK_SECONDS = 90

# The longest thing in any archive, rounded up. The Neuralink episode
# runs 8h38m, and Market Bubble shows go past four hours, so a stamp
# reading up to twelve is a plausible hour and anything above it is not.
_LONGEST_SHOW_HOURS = 12


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']{4,}", text.lower())}


def seconds(stamp: str) -> int:
    parts = [int(p) for p in stamp.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _as_written(total: int) -> str:
    """The second, written the way a person writes one.

    Keyed on whether there ARE hours, not on how the timestamp being
    replaced was punctuated. Copying the old format meant a correction
    from 1:02:00 back to five minutes printed "0:05:04", which is the
    right moment spelled like a machine.
    """
    h, m, s = total // 3600, total % 3600 // 60, total % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def stamped_lines(hits) -> list[tuple[int, str]]:
    """(second, text) for every line the model was shown.

    Unlabelled lines included: a line with no speaker prefix still has a
    timestamp, and half the archive has no labels. Dropping those would
    make this blind on exactly the episodes attribution cannot help with
    either.
    """
    out: list[tuple[int, str]] = []
    for hit in hits or ():
        for stamp, said in _STAMPED.findall(getattr(hit, "text_ts", "") or ""):
            try:
                out.append((seconds(stamp), said))
            except ValueError:
                continue
    return out


def readings(stamp: str) -> list[int]:
    """Every second a written stamp can honestly mean.

    "1:49" is 1m49s or 1h49m, and an answer about a three-hour show
    means the second. Nothing in the text says which, and guessing
    m:ss made the corrector treat four correct citations on one long
    episode as unsupported, because no line sits near second 109.
    Both readings are offered and any supported one vindicates it.

    Public, and shared, because this ambiguity belongs to every caller
    that reads a stamp out of an ANSWER rather than a transcript line.
    A stamp we generated is always full and means what it says; a stamp
    the model wrote is prose. Five verifiers were parsing model prose
    with the transcript rule, so on any episode past an hour they were
    checking figures, entities and quotes against the wrong second and
    reporting correct answers as unsupported.
    """
    out = [seconds(stamp)]
    if stamp.count(":") == 1:
        h, m = stamp.split(":")
        # Only where the hour reading is one a person could mean. "42:02"
        # is forty-two minutes; offering forty-two HOURS as an
        # alternative invents a second that no recording reaches, and a
        # window gathered there would vouch for anything.
        if int(h) <= _LONGEST_SHOW_HOURS and int(m) < 60:
            out.append(int(h) * 3600 + int(m) * 60)
    return out


def _supported(quote: str, lines: list[tuple[int, str]], at: int) -> bool:
    """Does a line near `at` carry this quote?

    The question that decides whether a citation drifted. Asking
    instead where the quote matches BEST is a different question with
    a worse answer: a host repeats a phrase, the far copy outscores
    the near one by a word, and a correct citation gets moved. Live,
    "a ton of alts did really well" sits at both 18:10 and 21:23; the
    answer cited 21:30 and was right.
    """
    target = _words(quote)
    if len(target) < _MIN_OVERLAP:
        return False
    return any(len(target & _words(said)) >= _MIN_OVERLAP
               for when, said in lines
               if abs(when - at) <= _SLACK_SECONDS)


def moment_of(quote: str, lines: list[tuple[int, str]]) -> int | None:
    """The second `quote` was said, if the passages establish it.

    None when nothing matches well enough, which is the common case: a
    paraphrase rather than a quotation. None means do not touch it.
    """
    if _NOT_A_QUOTE.search(quote):
        return None
    target = _words(quote)
    if len(target) < _MIN_OVERLAP:
        return None
    best, score = None, 0
    for at, said in lines:
        shared = len(target & _words(said))
        if shared > score:
            best, score = at, shared
    return best if score >= _MIN_OVERLAP else None


def correct(answer: str, hits) -> tuple[str, list[str]]:
    """Move any cited second the quotes contradict.

    Returns the answer and what changed, for the log. Safe to run on
    every reply: an answer with nothing to fix comes back unchanged.
    """
    if not answer or not hits:
        return answer, []
    lines = stamped_lines(hits)
    if not lines:
        return answer, []

    changes: list[str] = []
    # Reverse, so replacing one timestamp does not shift the offsets of
    # the ones still to check.
    for found in reversed(list(_QUOTED.finditer(answer))):
        truth = moment_of(found.group(1), lines)
        if truth is None:
            continue
        # The citation this quote hangs off is the nearest one before it.
        cited = list(_CITED.finditer(answer[:found.start()]))
        if not cited:
            continue
        mark = cited[-1]
        try:
            claims = readings(mark.group(1))
        except ValueError:
            continue
        # Leave it alone unless the cited moment carries nothing like
        # the quote, under either reading of the stamp.
        if any(_supported(found.group(1), lines, at) for at in claims):
            continue
        if any(abs(at - truth) <= _SLACK_SECONDS for at in claims):
            continue
        fixed = _as_written(truth)
        answer = answer[:mark.start(1)] + fixed + answer[mark.end(1):]
        changes.append(f"{mark.group(1)} -> {fixed} for "
                       f"{found.group(1)[:44]!r}")
    return answer, list(reversed(changes))
