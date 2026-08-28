"""Exact-token lookup, for the part of a query an embedding cannot carry.

A rare token is one word in four hundred. It barely moves the vector, so a
passage that literally contains it can rank below passages that are only
about the same subject. Three misses in one day came from that, every one
of them a thing the archive holds:

    "what did andre say"                -> Andrew Tate came back, while
                                           Andre from Grass sat in the index
    "what did mayne n ansem talk about" -> missed, though "what did mayne
                                           say" answers from his own episode
    "who made 54 million on the drop"   -> missed a line reading
                                           "54 million dollars on the drop"

What this does NOT do is decide anything. It contributes candidates to the
pool the reranker already scores, and the reranker still chooses the order
and the cut. A passage matched here that is not actually relevant is one
more thing for the reranker to reject, which it is better at than a
keyword rule would ever be. That is deliberate: retrieval quality is the
product, and this must only ever be able to add a right answer, never
remove one.

The index is built by scripts/build_term_index.py and shipped in the
image. It holds vector ids, not text, so it stays under a megabyte and
cannot drift from what the vectors say.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

INDEX = Path(__file__).resolve().parent.parent / "data" / "term_index.json"

# Same shape the builder uses, so a query is tokenised the way the corpus
# was. Any divergence here silently stops matches from ever being found.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9'.\-]{1,}")

# Words that would match half the corpus. The builder drops these by
# document frequency; this is the query side of the same idea, and it is
# short on purpose — the df cap does the real work.
_SKIP = frozenset("""
what which who whom whose when where why how did does do say says said talk
talked about discuss discussed think thinks mention mentioned the a an and or
but for with from that this they them their there here was were are is
""".split())


class TermIndex:
    """Vector ids for the rare tokens in a query.

    Loaded once. Missing or unreadable means every lookup returns nothing,
    which degrades to exactly the behaviour before this existed.
    """

    def __init__(self, path: Path = INDEX) -> None:
        self._ids: list[str] = []
        self._terms: dict[str, list[int]] = {}
        try:
            raw = json.loads(path.read_text())
            self._ids = raw.get("ids") or []
            self._terms = raw.get("terms") or {}
        except FileNotFoundError:
            logger.info("no term index at %s — exact matching is off", path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("term index unreadable (%s) — exact matching is "
                           "off, semantic search is unaffected", exc)

    def __len__(self) -> int:
        return len(self._terms)

    def _query_terms(self, query: str) -> list[str]:
        words = [w.strip(".-'") for w in _TOKEN.findall(query.lower())]
        found = [w for w in words
                 if len(w) >= 3 and w not in _SKIP and w in self._terms]
        # Numeric phrases: "54 million" is in one window, "54" is in dozens,
        # which is why the bare number is not in the index at all.
        for first, second in zip(words, words[1:], strict=False):
            if first and first[0].isdigit():
                phrase = f"{first} {second}"
                if phrase in self._terms:
                    found.append(phrase)
        return found

    def lookup(self, query: str, limit: int = 8) -> list[str]:
        """Vector ids for passages containing a rare token from the query.

        Rarest first: a token in two windows says far more about what was
        meant than one in fifty. Capped, because this only needs to put a
        few candidates in front of the reranker, not flood it.
        """
        if not self._terms:
            return []
        terms = self._query_terms(query)
        if not terms:
            return []
        terms.sort(key=lambda t: len(self._terms[t]))

        seen: list[str] = []
        taken: set[str] = set()
        for term in terms:
            for position in self._terms[term]:
                if position >= len(self._ids):
                    continue
                vector_id = self._ids[position]
                if vector_id not in taken:
                    taken.add(vector_id)
                    seen.append(vector_id)
                if len(seen) >= limit:
                    logger.debug("term index matched %s", terms[:3])
                    return seen
        return seen
