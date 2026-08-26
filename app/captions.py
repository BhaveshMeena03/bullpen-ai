"""WebVTT parsing — shared by the ingest script and the live clipper.

This lived in scripts/fetch_episodes.py, which is fine until something on
the server needs it too. The clipper captions a video from the same cues
the index was built from, and duplicating this parser would let the two
drift: the overlap-stripping below is subtle enough that a second copy
would quietly diverge and caption clips differently from how the
transcript reads.

Same reasoning as app/assets.py — logic used by both an offline script and
a live endpoint belongs where both can import it.
"""

from __future__ import annotations

import html
import re

_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
_TAG = re.compile(r"<[^>]+>")
_BLEEP = re.compile(r"\[\s*_+\s*\]")           # YouTube profanity bleep [ __ ]
_SPEAKER_MARK = re.compile(r"\s*>>+\s*")        # auto-caption speaker change
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = html.unescape(text)
    text = _BLEEP.sub("[expletive]", text)
    text = _SPEAKER_MARK.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _new_tail(prev_words: list[str], cue_words: list[str]) -> list[str]:
    """Return only the words in `cue_words` that aren't already covered by the
    overlap with the end of `prev_words`. YouTube auto-captions roll a window
    forward, so each cue repeats a prefix of already-emitted text; we find the
    largest k where the last k emitted words equal the first k cue words and
    keep only what follows."""
    max_overlap = min(len(prev_words), len(cue_words))
    for k in range(max_overlap, 0, -1):
        if prev_words[-k:] == cue_words[:k]:
            return cue_words[k:]
    return cue_words


def parse_vtt(vtt: str) -> list[dict]:
    """Parse a WebVTT file into deduped {t, text} segments, stripping YouTube's
    rolling-window overlap at word granularity."""
    segments: list[dict] = []
    emitted: list[str] = []  # trailing context for overlap detection
    for block in vtt.split("\n\n"):
        lines = block.strip().splitlines()
        ts_line = next((ln for ln in lines if "-->" in ln), None)
        if not ts_line:
            continue
        m = _TS.search(ts_line)
        if not m:
            continue
        start = _seconds(*m.groups())
        # The cue body is the lines AFTER the timestamp. (A WebVTT cue
        # identifier, if present, precedes the timestamp — never after it.)
        # Taking post-timestamp lines preserves fully-numeric spoken lines
        # like "2026" or "100" that a blanket isdigit() filter would drop —
        # which matters a lot for a finance podcast full of years and prices.
        ts_idx = lines.index(ts_line)
        body = _clean(" ".join(
            _TAG.sub("", ln) for ln in lines[ts_idx + 1:] if ln.strip()
        ))
        if not body:
            continue
        cue_words = body.split()
        new_words = _new_tail(emitted[-40:], cue_words)
        if not new_words:
            continue
        segments.append({"t": round(start, 2), "text": " ".join(new_words)})
        emitted.extend(new_words)
    return segments


def coalesce(segments: list[dict], min_gap: float = 6.0) -> list[dict]:
    """Merge tiny caption fragments into ~sentence-level segments so each
    carries a meaningful timestamp instead of two words."""
    out: list[dict] = []
    for seg in segments:
        if out and seg["t"] - out[-1]["t"] < min_gap:
            out[-1]["text"] = f'{out[-1]["text"]} {seg["text"]}'.strip()
        else:
            out.append(dict(seg))
    return out


def collapse_repeats(segments: list[dict], run_length: int = 3) -> list[dict]:
    """Drop Whisper's repetition loops.

    Whisper occasionally gets stuck and emits the same short line dozens of
    times in a row — one broadcast had "I know." thirty times inside two
    seconds. Across the transcribed broadcasts this was 1.7% of all
    segments, and 5.9% of one of them.

    It is not merely noise. Chunking packs consecutive segments into windows,
    so a long enough run produces a window that is almost entirely one
    repeated phrase: an index slot holding nothing, which retrieval can
    still return in place of something real.

    Only runs of `run_length` or more collapse, and only when consecutive
    and identical after normalising. Genuine repetition in speech — "yeah,
    yeah", "no, no, no" — is shorter than that and survives, which matters
    because it is sometimes the quotable part.
    """
    if not segments:
        return segments

    def key(seg: dict) -> str:
        return " ".join(str(seg.get("text", "")).lower().split())

    out: list[dict] = []
    i = 0
    while i < len(segments):
        j = i + 1
        while j < len(segments) and key(segments[j]) == key(segments[i]):
            j += 1
        run = j - i
        # Keep the first of a collapsed run: its timestamp is where the
        # speaker actually said the thing.
        out.append(segments[i])
        if run < run_length:
            out.extend(segments[i + 1:j])
        i = j
    return out
