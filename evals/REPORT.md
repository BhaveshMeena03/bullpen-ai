# Bullpen docs — coverage report

149 support questions run against the concierge, which answers only from the official documentation. Questions are written the way people type them in a support channel, not paraphrased out of the docs.

## Summary

| group | questions | answered | couldn't answer | rate |
|---|---:|---:|---:|---:|
| answerable | 70 | 69 | 1 | 99% |
| edge | 69 | 63 | 6 | 91% |
| guardrail (must refuse) | 10 | — | — | 10/10 refused |

## Guardrails

All 10 unsafe prompts were refused — seed-phrase requests, price predictions, and "what should I buy". The safety rules hold, so the coverage numbers below can be read at face value.

## Questions the documentation doesn't answer

6 questions a user would plausibly ask, with no grounded answer in the docs.

**Every question listed below is one the documentation could not answer.** Headings show how many of that topic's questions failed, out of how many were asked.

### Access & eligibility — 2 of 5 questions unanswered

- is bullpen available in the us
- is there an age requirement

### Limits & timing — 1 of 6 questions unanswered

- whats the max position size

### $ANSEM claim — 1 of 5 questions unanswered

- did i miss the ansem claim

### Competitions — 1 of 4 questions unanswered

- when does the current competition end

### Account recovery — 1 of 4 questions unanswered

- can i delete my account

## Retrieval failures

1 question that *is* covered by an indexed page but didn't retrieve one. This is the concierge's fault, not the documentation's.

- `can i change my email` — Account & wallet (low_confidence)

## Weakest sections

| section | unanswered | asked |
|---|---:|---:|
| Access & eligibility | 2 | 5 |
| Account & wallet | 1 | 10 |
| Limits & timing | 1 | 6 |
| $ANSEM claim | 1 | 5 |
| Competitions | 1 | 4 |
| Account recovery | 1 | 4 |
