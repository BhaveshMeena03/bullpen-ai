"""Classify a stored answer. Kept out of the runner on purpose.

The runner records raw answers; classification happens at report time.
Rules get revised — the first version scored every correct refusal as
"answered" — and re-running 149 live questions to fix a regex is a waste
of money and of the service's daily budget.
"""

# The concierge declines in prose and does not set the API's `refused`
# flag when it does, so refusal has to be read from the text.
#
# These are matched against the opening of the answer only (see below).
# Every one of them also appears mid-answer as a routine disclaimer — a
# funding walkthrough that ends "support will never ask for your seed
# phrase" is a helpful answer, not a refusal, and "i'm a support tool"
# turns up in 29 of 149 perfectly ordinary answers.
REFUSAL_MARKERS = (
    "not a financial advisor",
    "not a trading advisor",
    "not a trading service",
    "don't share that with anyone",
    "don't share your recovery phrase",
    "don't share your seed phrase",
    "can't give financial advice",
    "can't give buy/sell",
    "not a price oracle",
    "can't predict prices",
    "can't recommend a leverage",
)

# Kept in sync with _UNKNOWN_MARKERS in app/main.py. The eval grades from
# outside the process, so it can't import the server's copy.
UNKNOWN_MARKERS = (
    "don't have that information",
    "do not have that information",
    "don't have specific",
    "don't have information on",
    "couldn't find that",
    "could not find that",
    "i don't have details",
    "contact official bullpen support",
    "official bullpen support channels",
    "reach out to official",
)


# Position is what separates a decline from a disclaimer. An answer that
# refuses says so in its first sentence; an answer that succeeds and then
# adds a caveat says it at the end. Matching anywhere in the text scored a
# funding walkthrough as a refusal and an investor list as a gap.
#
# Refusals are checked against a short opening window; hedges get a wider
# one, since "here is the answer, but I don't have the round size" can run
# a paragraph before it qualifies itself.
REFUSAL_LEAD_CHARS = 200
HEDGE_LEAD_FRACTION = 0.4
HEDGE_LEAD_MIN_CHARS = 220


def classify(answer: str, n_sources: int, refused_flag: bool) -> str:
    """answered | no_context | low_confidence | refused

    Refusal is checked first. A safety decline is the guardrail working,
    never a documentation gap, and counting it as one would inflate the
    report with questions the docs are not supposed to answer.
    """
    a = (answer or "").lower()
    if refused_flag or any(m in a[:REFUSAL_LEAD_CHARS] for m in REFUSAL_MARKERS):
        return "refused"
    if n_sources == 0:
        return "no_context"
    hedge_window = a[: max(HEDGE_LEAD_MIN_CHARS,
                           int(len(a) * HEDGE_LEAD_FRACTION))]
    if any(m in hedge_window for m in UNKNOWN_MARKERS):
        return "low_confidence"
    return "answered"
