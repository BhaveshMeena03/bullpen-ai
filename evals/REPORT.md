# Bullpen docs — coverage report

149 support questions run against the concierge, which answers only from the official documentation. Questions are written the way people type them in a support channel, not paraphrased out of the docs.

## Summary

| group | questions | answered | couldn't answer | rate |
|---|---:|---:|---:|---:|
| answerable | 70 | 69 | 1 | 99% |
| edge | 69 | 59 | 10 | 86% |
| guardrail (must refuse) | 10 | — | — | 10/10 refused |

## Guardrails

All 10 unsafe prompts were refused — seed-phrase requests, price predictions, and "what should I buy". The safety rules hold, so the coverage numbers below can be read at face value.

## Questions the documentation doesn't answer

10 questions a user would plausibly ask, with no grounded answer in the docs. Grouped by the section that would own them.

### Access & eligibility — 3 of 5 unanswered

- is bullpen available in the us
- do i need kyc
- is there an age requirement

### Limits & timing — 1 of 6 unanswered

- how many wallets can i track at once

### $ANSEM claim — 1 of 5 unanswered

- did i miss the ansem claim

### Security — 1 of 5 unanswered

- is bullpen audited

### Tax & records — 1 of 3 unanswered

- can i export my trade history

### Rewards mechanics — 1 of 7 unanswered

- how much do i earn per referral

### Competitions — 1 of 4 unanswered

- when does the current competition end

### Account recovery — 1 of 4 unanswered

- i lost access to my email

## Retrieval failures

1 questions that *are* covered by an indexed page but didn't retrieve one. These are the concierge's fault, not the documentation's.

- `who invested in bullpen` — Platform detail (low_confidence)

## Weakest sections

| section | unanswered | asked |
|---|---:|---:|
| Access & eligibility | 3 | 5 |
| Platform detail | 1 | 8 |
| Limits & timing | 1 | 6 |
| $ANSEM claim | 1 | 5 |
| Security | 1 | 5 |
| Tax & records | 1 | 3 |
| Rewards mechanics | 1 | 7 |
| Competitions | 1 | 4 |
| Account recovery | 1 | 4 |
