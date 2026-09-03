"""Refuse to name the wrong host, rather than guess which one spoke.

A reply carrying both hosts' words is normal -- it is a conversation, and
they talk over each other for four hours. What is not normal is one man's
sentence in the other's mouth:

    reply:       "Ansem said 'I own none of the token. I have been
                  sidelined.'"
    transcript:  FaZe Banks said it, 0:40:17

That is the $ANSEM token. Banks saying he owns none of it is a disclosure
of non-involvement, and the reason he repeats it on air. Ansem saying it
about his own coin is a different claim, and a false one. The quotation
marks are what a reader screenshots.

The prompt already forbids this. Rule 5b describes this precise failure,
naming the Solana case by name, and rules 5, 5a, 5c, 5d, 5e and 5f were
added around it over one night. Measured across a hundred questions the
residual is about 2%: two replies in a hundred still credit a quote to
the host the QUESTION named rather than the one the transcript labels.
Seven rules have not closed it, so this closes it in code.

Deliberately narrow. It does not correct the name -- picking the other
host would be a second guess, and a wrong correction is worse than a
vague one. It demotes to the phrasing rule 5 already asks for when the
speaker is not established:

    "Ansem said" -> "one of the hosts said"

Nothing else changes. The quote, the timestamp, the episode and the link
are untouched, because those were right. Worst case a reply is slightly
vaguer than it needed to be; best case it stops being false about a real
person.
"""

from __future__ import annotations

import re

# Hosts, longest first, so "FaZe Banks" is matched before "Banks".
HOSTS = ("FaZe Banks", "Ansem", "Banks")

# Only the two hosts are voice-labelled. Every guest line is unprefixed,
# so this must never touch a guest's name: doing so would take "what did
# Jesse say about Base" from a good answer to a vague one.
_CANON = {"faze banks": "FaZe Banks", "banks": "FaZe Banks",
          "ansem": "Ansem"}

# "Ansem said", "Banks noted", "FaZe Banks explained" -- and the
# possessive and colon forms the model also produces.
# The gap must not cross a comma or a pipe, and the name must not be part of
# an episode title. Both come from the same live failure: "in the Ansem
# Edition episode, FaZe Banks mentions..." matched Ansem here, ran the gap
# across the comma to reach "mentions", and rewrote the TITLE — the reply
# went out reading "in the one of the hosts Edition episode".
#
# A real attribution does not need a comma to reach its verb: "Ansem said",
# "Ansem explained around 3:05". Anything that has to cross one is reaching
# past the sentence it belongs to.
_CREDIT = re.compile(
    r"\b(FaZe Banks|Ansem|Banks)\b"
    r"(?!\s+(?:Edition|Episode|Ep\b))"
    r"(?P<gap>[^.!?\n\",|]{0,60}?)"
    r"\b(?P<verb>said|says|noted|notes|explained|explains|argued|argues"
    r"|described|describes|recalled|recalls|admitted|admits|mentioned"
    r"|mentions|claimed|claims|stated|states|revealed|reveals|put it"
    r"|called|calls|added|adds|joked|jokes|pointed out)\b")

# A line inside an excerpt: "[12:02] FaZe Banks: I put close to seven..."
_LABELLED = re.compile(r"^\[[\d:]+\]\s*([A-Z][A-Za-z ]{2,20}):\s*(.+)$", re.M)

_QUOTED = re.compile(r'"([^"\n]{18,200})"')

# Enough shared vocabulary to be confident the quote came from that line
# rather than merely resembling it.
_MIN_OVERLAP = 3


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']{4,}", text.lower())}


def labelled_lines(hits) -> list[tuple[str, str]]:
    """(speaker, text) for every labelled line the model was shown."""
    out: list[tuple[str, str]] = []
    for hit in hits or ():
        stamped = getattr(hit, "text_ts", "") or ""
        for who, said in _LABELLED.findall(stamped):
            canon = _CANON.get(who.strip().lower())
            if canon:
                out.append((canon, said))
    return out


def speaker_of(quote: str, lines: list[tuple[str, str]]) -> str | None:
    """Which host said `quote`, if the passages establish it.

    Returns None when no line matches well enough, which is the common
    case: a paraphrase, or a guest speaking on an unlabelled line. None
    means "do not touch it" -- this only ever acts on a disagreement it
    can actually demonstrate.
    """
    target = _words(quote)
    if len(target) < 4:
        return None
    best, score = None, 0
    for who, said in lines:
        shared = len(target & _words(said))
        if shared > score:
            best, score = who, shared
    return best if score >= _MIN_OVERLAP else None


def _same(a: str, b: str) -> bool:
    return _CANON.get(a.lower().strip()) == _CANON.get(b.lower().strip())


def correct(answer: str, hits) -> tuple[str, list[str]]:
    """Demote any host credit the passages contradict.

    Returns the answer and a list of what changed, for the log. An answer
    with nothing to fix comes back identical, by identity, so this is
    safe to run on every reply.
    """
    if not answer or not hits:
        return answer, []
    lines = labelled_lines(hits)
    if not lines:
        return answer, []

    changes: list[str] = []
    # Walk quotes in reverse so earlier offsets stay valid as we edit.
    for found in reversed(list(_QUOTED.finditer(answer))):
        quote = found.group(1)
        truth = speaker_of(quote, lines)
        if not truth:
            continue
        # The nearest credit BEFORE this quote is the one it belongs to.
        credits = list(_CREDIT.finditer(answer[:found.start()]))
        if not credits:
            continue
        credit = credits[-1]
        claimed = credit.group(1)
        if _same(claimed, truth):
            continue
        # Replace only the name, leaving the verb and everything between
        # it intact: "Ansem explained around 3:05" keeps its shape.
        start, end = credit.start(1), credit.end(1)
        answer = answer[:start] + "one of the hosts" + answer[end:]
        changes.append(f"{claimed!r} -> 'one of the hosts' for "
                       f"{quote[:44]!r} (labelled {truth})")
    return answer, list(reversed(changes))
