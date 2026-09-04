"""Spell the names right, even when the captions do not.

Whisper mishears the words this archive is most about. Measured across
every transcript:

    Solana      → "Salana"        430 of 1,131   (39%)
    Polymarket  → "poly market"   151 of 268     (56%)
    Hyperliquid → "Hyperlid"      149, plus "hyper liquid" 80  (54%)
    pump.fun    → "pumpfun"        43 of 44      (98%)
    friend.tech → "Frentech"       14 of 14     (100%)
    Mt Gox      → "Mount Gox"       9 of 9      (100%)

Two consequences, and the second is the one people see.

Retrieval mostly survives it -- the embeddings match on meaning, which is
why a search for "friend.tech" finds "Frentech" and a search for Mt Gox
found "Mount Gox". What cannot survive it is the exact-token index, which
matches letters and so has holes in precisely the places Whisper failed.

And when the model quotes a mangled line, the error goes out under a real
person's name: "Hyperlid briefly flipped Salana price" was posted as
FaZe Banks' words. He said Hyperliquid and Solana. Reproducing the
transcription error IS the misquote, so correcting it is the faithful
thing rather than the liberty.

Only unambiguous manglings are here. "per" is 170 hits and almost always
the English word rather than PURR; "Seoul" is 13 hits and a real city as
often as it is SOL. Both are left alone -- a wrong correction is worse
than an uncorrected error, because it is one this code chose.
"""

from __future__ import annotations

import re

# mangled form -> what was actually said. Ordered longest-first at
# compile time so "hyper liquid" is tried before "hyperlid".
_ALIASES: dict[str, str] = {
    "salana": "Solana",
    "salada": "Solana",
    "salona": "Solana",
    "hyperlid": "Hyperliquid",
    "hyper liquid": "Hyperliquid",
    "poly market": "Polymarket",
    # Counted in the transcripts before adding: 285 lines say "robin
    # hood", 126 say "pump fun", 35 say "micro strategy". Every one of
    # them was unreachable by exact search, because expand() only knows
    # the spellings listed here -- a search for "Robinhood" matched the
    # 129 lines that spell it correctly and missed the other 285.
    "robin hood": "Robinhood",
    "pump fun": "pump.fun",
    "micro strategy": "MicroStrategy",
    "pumpfun": "pump.fun",
    "frentech": "friend.tech",
    "fren tech": "friend.tech",
    "z cash": "Zcash",
    "mount gox": "Mt. Gox",
    "board ape": "Bored Ape",
    "corweave": "CoreWeave",
    "lucanets": "Luca Netz",
    "bull penn": "Bullpen",
}

_ALTERNATION = "|".join(
    re.escape(k) for k in sorted(_ALIASES, key=len, reverse=True))
_PATTERN = re.compile(rf"\b(?:{_ALTERNATION})\b", re.IGNORECASE)


def _cased(mangled: str, correct: str) -> str:
    """Keep SHOUTING when the source was shouting, otherwise use the
    canonical spelling. A transcript in caps should not force "SOLANA"
    into prose, but neither should it be silently lowercased."""
    return correct.upper() if mangled.isupper() and len(mangled) > 3 else correct


def fix(text: str) -> tuple[str, list[str]]:
    """Correct known caption manglings.

    Returns the text and what was changed, for the log. Text with nothing
    to fix comes back unchanged, so this is safe to run on everything.
    """
    if not text:
        return text, []
    changed: list[str] = []

    def swap(found: re.Match) -> str:
        was = found.group(0)
        right = _cased(was, _ALIASES[was.lower()])
        if was != right:
            changed.append(f"{was!r}->{right!r}")
        return right

    return _PATTERN.sub(swap, text), changed


def expand(query: str) -> list[str]:
    """The query plus the spellings the transcripts actually use.

    For the exact-token index, which matches letters rather than meaning
    and therefore misses every line Whisper got wrong. Somebody searching
    "solana" should still reach the four hundred lines that say "Salana".
    """
    lower = (query or "").lower()
    out = []
    for mangled, correct in _ALIASES.items():
        if correct.lower() in lower and mangled not in lower:
            out.append(re.sub(re.escape(correct), mangled, query,
                              flags=re.IGNORECASE))
    return out
