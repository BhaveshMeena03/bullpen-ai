"""Classify a stored answer. Kept out of the runner on purpose.

The runner records raw answers; classification happens at report time.
Rules get revised — the first version scored every correct refusal as
"answered" — and re-running 149 live questions to fix a regex is a waste
of money and of the service's daily budget.
"""

# The concierge declines in prose and does not set the API's `refused`
# flag when it does, so refusal has to be read from the text. Phrases are
# chosen to be things a grounded documentation answer would never say.
REFUSAL_MARKERS = (
    "not a financial advisor",
    "not a trading advisor",
    "i'm a support tool",
    "i am a support tool",
    "don't share that with anyone",
    "don't share your recovery phrase",
    "don't share your seed phrase",
    "never ask for it",
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


def classify(answer: str, n_sources: int, refused_flag: bool) -> str:
    """answered | no_context | low_confidence | refused

    Refusal is checked first. A safety decline is the guardrail working,
    never a documentation gap, and counting it as one would inflate the
    report with questions the docs are not supposed to answer.
    """
    a = (answer or "").lower()
    if refused_flag or any(m in a for m in REFUSAL_MARKERS):
        return "refused"
    if n_sources == 0:
        return "no_context"
    if any(m in a for m in UNKNOWN_MARKERS):
        return "low_confidence"
    return "answered"
