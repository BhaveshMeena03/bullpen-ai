"""Delete the denial an answer then contradicts.

    "I couldn't find that in the episodes I've indexed. The excerpts
     mention Michael Cat repeatedly — described as the head of
     production at Market Bubble around 3:05."

The first sentence says no. The rest answers the question. On X a reader
takes the first line and scrolls, so a reply that worked reads as a tool
that did not -- and that is the reply somebody screenshots when deciding
whether this thing is worth anything.

Rule 1a was written at this twice: once as a principle, then again as a
mechanic listing the forbidden openings verbatim. It went 4-in-15 to
2-in-15 and stopped. Every other thing that resisted the prompt this way
-- the joke pool, model-written links, quotes credited to the wrong host
-- was fixed by code in the end. This is that.

Narrow on purpose. It only ever removes a denial that the SAME answer
goes on to contradict with a citation, so:

  - a real refusal, which cites nothing, is untouched. "I couldn't find
    that in the episodes I've indexed" is the honest answer to a question
    the archive cannot answer, and it has to survive intact.
  - a qualifier that comes AFTER the answer is untouched. "Around 27:09
    ... though he doesn't put it in those words" is precision, and rule
    1a explicitly asks for it.
"""

from __future__ import annotations

import re

# A citation, which is what makes the denial false. The prompt requires
# every real answer to name a moment, so this is the mark of one.
_CITES = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

# How the denials actually open. Collected from live replies rather than
# imagined: each of these was posted.
_DENIAL = re.compile(
    r"""(?ix)^\W*
    (?: i \s+ (?: couldn't | could \s+ not | can't | cannot | don't | do \s+ not
                | didn't | did \s+ not )
        \s+ (?: find | see | have )
      | i \s+ looked ,? \s+ and \s+ (?:couldn't|could \s+ not)
      | (?: there \s+ (?:is|are) \s+ )? no \s+
        (?: direct | specific | clear | exact )? \s*
        (?: statement | mention | quote | passage | excerpt | record )
      | not \s+ (?: in \s+ those \s+ words | word \s+ for \s+ word
                  | that \s+ exact | exactly | quite )
      | nothing \s+ (?: in | matching | specific )
      | the \s+ (?:excerpts|transcripts) \s+ (?:don't|do \s+ not)
    )\b""")

# Where a denial hands over to the answer inside one sentence: "Not in
# those words, but the archive has this: ..."
_PIVOT = re.compile(
    r"(?i)[,;—-]?\s*\b(?:but|however|though|although|that said)\b\s*,?\s+")

# Connectors that open the surviving sentence and read oddly once the
# denial in front of them is gone.
_ORPHAN = re.compile(
    r"(?i)^\W*(?:however|but|that said|what i can see is that|"
    r"what i did find (?:is|was)|from what'?s shown)\s*[,:]?\s+")

_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Below this the remainder is a fragment rather than an answer. Set at
# 40 rather than 60 because "Around 3:04:39 he laid out the levels he
# would bid into." is 55 characters and a complete answer -- the length
# guard is only there to stop a bare "Around 27:09." being left behind,
# and the citation requirement already does most of that work.
_MIN_REMAINDER = 40


def _tidy(text: str) -> str:
    text = _ORPHAN.sub("", text.strip())
    if text and text[0].islower() and not text.startswith(("'", '"')):
        text = text[0].upper() + text[1:]
    return text


def strip_denial(answer: str) -> tuple[str, str | None]:
    """Remove a leading denial the answer itself disproves.

    Returns the answer and the removed text, or None when nothing was
    removed. An answer with nothing to fix comes back unchanged, so this
    is safe to run on everything.
    """
    text = (answer or "").strip()
    if not text or not _CITES.search(text):
        # Nothing cited: either a real refusal or not an answer at all.
        return answer, None

    first = _SENTENCE.split(text, maxsplit=1)
    head = first[0]
    rest = first[1] if len(first) > 1 else ""

    if _DENIAL.search(head):
        # The denial is its own sentence, and the answer follows it.
        if len(rest.strip()) >= _MIN_REMAINDER and _CITES.search(rest):
            return _tidy(rest), head.strip()
        # Or the denial opens a sentence that pivots mid-way: "Not in
        # those words, but around 27:09 ..."
        pivot = _PIVOT.search(head)
        if pivot:
            tail = (head[pivot.end():] + " " + rest).strip()
            if len(tail) >= _MIN_REMAINDER and _CITES.search(tail):
                return _tidy(tail), head[:pivot.start()].strip()
    return answer, None
