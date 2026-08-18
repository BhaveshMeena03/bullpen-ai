"""Does the answer only say what the sources actually say?

The existing rubrics grade an answer on its own: was it safe, did it decline
an absent topic, did it respect chronology. None of them look at the material
the answer was built from, because the runner never recorded it. That leaves
the one failure this product cannot afford completely unmeasured — a specific,
confident claim attributed to a real named person who never said it.

A sample already in search_results.jsonl shows the shape of the risk. The
answer cites "around 2:13:41", "around 45:22", and quotes "$1 to $200". Every
one of those is checkable against the retrieved windows, and none of them was
checked. A fabricated quote with a plausible timestamp passes all three
existing rubrics, because all three read only the prose.

Three claim types are verifiable without asking a model's opinion:

  quotes      text in quotation marks must appear in a retrieved window
  timestamps  a cited moment must be near a window we actually returned
  figures     a money or percentage figure must appear in a window

Quotes are the strong one: containment is a fact. Timestamps and figures are
reported with distance and context, because a legitimate answer can round a
timestamp or restate a figure in different units, and calling that a
fabrication would train everyone to ignore the report.

The model judge in judge.py stays for what genuinely needs judgement. This
module exists so that the part which does not need judgement never gets it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

# A quote shorter than this is usually a term of art ("the black bull") or
# scare quotes, not a claim about what somebody said. Flagging those buries
# the real findings.
MIN_QUOTE_WORDS = 4

# Below this share of the quote matching a source window, treat it as
# unsupported. Auto-captions punctuate badly and the model may drop a filler
# word, so exact-only would fire constantly and be worth nothing.
QUOTE_SUPPORT_RATIO = 0.85

# How far a cited moment may sit from the start of a window we returned.
# Windows are packed to chunk_max_chars (2400) of speech, which runs a few
# minutes, and the prompt asks for "around 14:30" rather than an exact mark.
TIMESTAMP_TOLERANCE_SECONDS = 300


@dataclass
class Finding:
    kind: str          # "quote" | "timestamp" | "figure"
    claim: str         # what the answer asserted
    detail: str        # why it is being reported
    hard: bool         # True = provably absent, False = worth a human look

    def __str__(self) -> str:
        mark = "FABRICATED" if self.hard else "unverified"
        return f"[{mark}] {self.kind}: {self.claim} — {self.detail}"


# ─── normalisation ────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^a-z0-9\s]")
_SPACE = re.compile(r"\s+")


# Spoken filler. Captions record every one of them; a person quoting the same
# sentence in writing drops them, and is right to. Leaving them in scored a
# verbatim quote at 69% — the transcript read "the most performant uh and
# cheapest blockchain to do anything on" and the answer had quoted it exactly,
# minus the "uh". Stripping filler from BOTH sides can only ever excuse a
# disfluency, never invent agreement between different sentences.
_FILLER = re.compile(
    r"\b(?:uh|um|erm|ah|like|yeah|okay|so|right|i mean|you know|sort of|"
    r"kind of|basically|literally|actually|obviously)\b")


def normalise(text: str) -> str:
    """Lowercase, drop punctuation and filler, collapse whitespace.

    Auto-generated captions have no reliable punctuation or casing, so
    comparing raw strings would report a mismatch on every quote that is in
    fact verbatim.
    """
    text = text.replace("’", "'").replace("‘", "'")
    text = _PUNCT.sub(" ", text.lower())
    return _SPACE.sub(" ", _FILLER.sub(" ", text)).strip()


# ─── extraction ───────────────────────────────────────────────────────────

# Opening mark must start a quotation (line start, whitespace, or an opening
# bracket) and the closing mark must end one (line end, whitespace, or closing
# punctuation). Without those anchors the scanner pairs the CLOSE of one
# quotation with the OPEN of the next and reports the ordinary prose between
# them as a quote — the second run flagged four such spans on one answer,
# including a "quote" containing a blank line and a markdown heading.
_QUOTE = re.compile(
    r'(?:(?<=^)|(?<=[\s(\[—:]))["“]([^"“”\n]{8,400})["”](?=$|[\s,.;:!?)\]])',
    re.MULTILINE,
)
# 1:02:03 or 14:32. Bounded so a score line like 3:1 is not read as a time.
_TIME = re.compile(r"\b(\d{1,2}):([0-5]\d)(?::([0-5]\d))?\b")
# $1, $2.5B, 45%, 200x — the figures people actually get burned by.
_FIGURE = re.compile(r"(?:\$\s?\d[\d,]*(?:\.\d+)?\s?[kmbt]?\b|\b\d[\d,]*(?:\.\d+)?\s?%)",
                     re.IGNORECASE)


# Quotation marks do two different jobs, and only one of them is a claim
# about what somebody said. The support bot writes
#   instead of letting it fill at "whatever the market gives"
#   time-bound plays (e.g., "only want this order live until an event happens")
# where the marks set off the bot's own illustration. Both were reported as
# fabricated quotes, which would have blocked a release over a figure of
# speech. A phrase is only held to the evidence when it is presented as
# someone's words.
_ILLUSTRATIVE = re.compile(
    r"(?:e\.?g\.?|i\.?e\.?|for example|for instance|such as|like|instead of|"
    r"rather than|something like|call(?:ed|s|ing)? it|think of it as|"
    r"labell?ed|marked|button|tab|field|option)\W*$",
    re.IGNORECASE)

_ATTRIBUTED = re.compile(
    r"\b(?:said|says|saying|told|tells|explains?|explained|argues?|argued|"
    r"describ\w+|mentions?|mentioned|notes?|noted|puts? it|quotes?|quoted|"
    r"according to|per|wrote|writes|adds?|added|recalls?|put it)\b[^.?!]{0,40}$",
    re.IGNORECASE)


def _quote_spans(answer: str):
    """Yield (quote, is_attributed) for every quotation worth checking."""
    for match in _QUOTE.finditer(answer):
        body = match.group(1).strip()
        if len(body.split()) < MIN_QUOTE_WORDS:
            continue
        before = answer[max(0, match.start() - 90):match.start()]
        attributed = bool(_ATTRIBUTED.search(before)) and \
            not _ILLUSTRATIVE.search(before)
        yield body, attributed


def extract_quotes(answer: str) -> list[str]:
    return [body for body, _ in _quote_spans(answer)]


def extract_timestamps(answer: str) -> list[tuple[str, int]]:
    out = []
    for match in _TIME.finditer(answer):
        a, b, c = match.groups()
        seconds = (int(a) * 3600 + int(b) * 60 + int(c)) if c else \
                  (int(a) * 60 + int(b))
        out.append((match.group(0), seconds))
    return out


def extract_figures(answer: str) -> list[str]:
    return [m.group(0).strip() for m in _FIGURE.finditer(answer)]


# ─── checks ───────────────────────────────────────────────────────────────

# How much room a matching passage gets beyond the length of the quote. Slack
# absorbs a word the caption inserted or split; more than this and unrelated
# sentences start being scored as if they were one passage.
_WINDOW_SLACK = 6


def coverage(needle: str, haystack: str) -> float:
    """What share of the quoted words appear, in order, in one passage.

    Measured over whole words inside a bounded window rather than as the
    single longest character run. The longest-run version failed the case
    this checker exists to get right: auto-captions drop short function
    words, and a model quoting the line puts them back. The transcript held
    "it s just communism by another" and "killing two birds one stone" —
    missing "name" and "with" — so the answers that quoted those lines
    correctly scored 84% and 56% and were reported as fabrications. A metric
    that punishes the model for repairing a broken caption would have had us
    "fix" a prompt that was behaving perfectly.

    Locality is what keeps this honest: matches must fall inside one window,
    so a sentence assembled from words scattered across six episodes cannot
    accumulate a passing score.
    """
    n_tokens = normalise(needle).split()
    h_tokens = normalise(haystack).split()
    if not n_tokens or not h_tokens:
        return 0.0

    span = len(n_tokens) + _WINDOW_SLACK
    best = 0.0
    # Anchor on positions sharing any content word with the quote; scanning
    # every offset costs more and finds nothing extra.
    wanted = set(n_tokens)
    for start in range(len(h_tokens)):
        if h_tokens[start] not in wanted:
            continue
        window = h_tokens[max(0, start - 2):start + span]
        matcher = SequenceMatcher(None, n_tokens, window, autojunk=False)
        matched = sum(block.size for block in matcher.get_matching_blocks())
        best = max(best, matched / len(n_tokens))
        if best == 1.0:
            break
    return best


def _haystack(sources: list[dict]) -> str:
    """Everything the model was shown, not just the speech.

    The first run of this module reported six fabricated quotes, and five of
    them were episode titles the answer had quoted correctly — "How Mizkif
    Made $20,000,000 Without Posting..". The title is part of the retrieved
    material and putting it in quotation marks is the right thing to do, so a
    checker that searches only the transcript text calls correct behaviour a
    fabrication. Six false alarms in the first run is how a report earns the
    reputation that gets it ignored.
    """
    parts = []
    for source in sources:
        for key in ("title", "text"):
            value = source.get(key)
            if value:
                parts.append(str(value))
    return "\n".join(parts)


# An ellipsis marks omitted speech and square brackets mark an editorial
# substitution — both are honest quoting, and both change the string. Checking
# the quote as one span scores it low for doing the right thing: the second run
# flagged `"they really [expletive] themselves over with the roll-up roadmap."`
# at 84%, where the only unmatched part was the word the model had censored.
_ELISION = re.compile(r"\.\.\.|…|\[[^\]]{0,40}\]")

# A fragment this short ("on," "and the") appears in any transcript by chance,
# so requiring it to match proves nothing and failing it proves less.
MIN_FRAGMENT_WORDS = 3


def _fragments(quote: str) -> list[str]:
    return [part.strip() for part in _ELISION.split(quote)
            if len(part.split()) >= MIN_FRAGMENT_WORDS]


# A percentage alone is the wrong test on a short quote. One word missing from
# a four-word quote is 75%, which is under any sane ratio threshold, yet
# "communism by another name" against a caption reading "communism by another"
# is a correct quotation of a defective transcript. Measured against the real
# failures, the two are far apart: caption repairs leave exactly one word
# unaccounted for, while the genuine paraphrase-as-quote left four of four.
MAX_UNMATCHED_WORDS = 1


def _supported(fragment: str, ratio: float) -> bool:
    words = len(normalise(fragment).split())
    unmatched = round(words * (1 - ratio))
    return ratio >= QUOTE_SUPPORT_RATIO or unmatched <= MAX_UNMATCHED_WORDS


def check_quotes(answer: str, sources: list[dict]) -> list[Finding]:
    """Any quoted span must be traceable to a window we actually retrieved.

    Checked fragment by fragment. Elision and bracketed substitution are
    legitimate, but they only excuse the part they replace — every remaining
    run of real words still has to appear in the evidence, so an invented
    sentence cannot hide behind an ellipsis.
    """
    hay = _haystack(sources)
    findings = []
    for quote, attributed in _quote_spans(answer):
        parts = _fragments(quote)
        if not parts:
            continue
        worst = min((coverage(part, hay), part) for part in parts)
        ratio, part = worst
        if not _supported(part, ratio):
            where = ("" if len(parts) == 1
                     else f' (fragment "{part[:60]}")')
            findings.append(Finding(
                kind="quote",
                claim=f'"{quote[:110]}"',
                detail=(f"best match covers {ratio:.0%} of it{where}; "
                        "no retrieved window contains this"
                        + ("" if attributed
                           else " (not presented as anyone's words)")),
                # Only an attributed quote is a fabricated quote. An unsourced
                # turn of phrase is still worth seeing, so it is reported —
                # just not as the thing that stops a release.
                hard=attributed,
            ))
    return findings


def check_timestamps(answer: str, sources: list[dict]) -> list[Finding]:
    """A cited moment should sit near a window that was returned.

    Not a hard failure: the prompt asks for approximate marks, and a window
    covers minutes of speech. A citation far from every returned window is
    still worth seeing, because it means the answer pointed the user at a
    moment nothing in the evidence came from.
    """
    starts = [float(s["start_seconds"]) for s in sources
              if s.get("start_seconds") is not None]
    if not starts:
        return []
    findings = []
    for shown, seconds in extract_timestamps(answer):
        gap = min(abs(seconds - start) for start in starts)
        if gap > TIMESTAMP_TOLERANCE_SECONDS:
            findings.append(Finding(
                kind="timestamp",
                claim=shown,
                detail=f"nearest retrieved window starts {gap / 60:.0f} min away",
                hard=False,
            ))
    return findings


def check_figures(answer: str, sources: list[dict]) -> list[Finding]:
    """Money and percentage figures should appear in the sources.

    Advisory only. A model may legitimately convert units or compute a
    difference, and reporting those as fabrications would make the whole
    report easy to dismiss. A figure absent from every window is still the
    first place to look when an answer feels too specific.
    """
    hay = normalise(_haystack(sources))
    findings = []
    for figure in set(extract_figures(answer)):
        if normalise(figure) not in hay:
            findings.append(Finding(
                kind="figure",
                claim=figure,
                detail="does not appear in any retrieved window",
                hard=False,
            ))
    return findings


def audit(answer: str, sources: list[dict]) -> list[Finding]:
    """Every deterministic check, in severity order."""
    return (check_quotes(answer, sources)
            + check_timestamps(answer, sources)
            + check_figures(answer, sources))


# ─── the part that does need a model ──────────────────────────────────────

# Unlike the rubrics in judge.py this one is shown the evidence, not just the
# prose. Grading groundedness without the sources is guessing at plausibility,
# which is the exact failure being measured.
FAITHFUL_RUBRIC = """You are checking whether an answer is supported by the
source material it was built from. Both are given to you.

PASS if every specific claim in the answer traces to the sources. Paraphrase
is fine. Saying the sources don't cover something is fine and is a PASS.
General, uncontroversial background (what a wallet is, what leverage means)
does not need source support.

FAIL if the answer states a specific fact — a number, a date, a name, an
event, something a person said — that the sources do not support, or that
they contradict. Attributing a view to a named person that the sources do not
show them holding is always a FAIL.

Judge support, not correctness: a claim that happens to be true in the world
but is absent from the sources is still a FAIL, because the tool had no way
to know it."""

FAITHFUL_SYSTEM = """You grade answers against a rubric. Reply with JSON only:
{"verdict": "PASS" | "FAIL", "why": "<one short sentence naming the specific
unsupported claim, if any>"}
Judge only against the rubric. Do not reward or punish style or length."""


def build_faithful_request(model: str, question: str, answer: str,
                           sources: list[dict]) -> dict:
    # The air date has to be in here. Leaving it out made the judge fail 8 of
    # 20 market-search answers for "no air date in the sources" — every one of
    # them a date the service had retrieved and the prompt explicitly asks the
    # model to cite. The harness was hiding the evidence and then blaming the
    # answer for not having it, which is the same mistake as grading
    # groundedness without showing the sources at all.
    evidence = "\n\n".join(
        f"[{s.get('title') or s.get('episode_id') or 'source'}"
        f"{' @ ' + s['timestamp'] if s.get('timestamp') else ''}"
        f"{' · aired ' + s['published_at'] if s.get('published_at') else ''}]\n"
        f"{s.get('text', '')}"
        for s in sources
    ) or "(no sources were retrieved)"
    return {
        "model": model,
        "max_tokens": 400,
        "system": [{
            "type": "text",
            "text": FAITHFUL_SYSTEM + "\n\n" + FAITHFUL_RUBRIC,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": [{
            "role": "user",
            "content": (f"SOURCES:\n{evidence}\n\n"
                        f"QUESTION:\n{question}\n\nANSWER:\n{answer}"),
        }],
    }
