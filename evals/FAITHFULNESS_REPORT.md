# Faithfulness — do answers stay inside their sources?

Each answer is checked against the exact material it was built from. Quotes are verified by containment, which is a fact about the output; timestamps and figures are reported with context, because rounding and unit changes are legitimate.

| surface | questions | fabricated quotes | unverified | judged unsupported |
|---|---:|---:|---:|---:|
| market search | 12 | 0 | 0 | 5 |
| support bot | 12 | 0 | 0 | 1 |

A non-zero **fabricated quotes** column is a release blocker: it means the tool put words in someone's mouth. The other two columns are for reading, not for gating.

### Market search

- `what did the helius guy say about solana`
  - judge: The answer attributes a specific claim about doubling the inflation rate to Helius/the CEO, but the source says they're 'pushing a proposal to double this inflation rate' (i.e., halve it, not increase
- `the episode about someone suing the sec`
  - judge: The answer attributes claims to Gary Gensler (misspelled as 'Gendler' in sources) about allegedly being incentivized by Elizabeth Warren for a Treasury Secretary position, but the sources present thes
- `which stocks are the robot play`
  - judge: The answer claims Take-Two Interactive is 'a robot play,' but the sources show Take-Two is a gaming company being discussed for its digital IP and game sales, with no mention of robotics involvement.
- `mizkif giving his chat money to trade`
  - judge: The answer states Mizkif is giving his chat '$300,000 to invest' but the sources show he stated '$300,000' as a past amount and later reveals the portfolio reached '$406,000', which he attributes part
- `selling a crypto company to kraken`
  - judge: The answer states the acquisition occurred in 'September 2025,' but the sources only specify 'September' without a year, and the episode aired in May 2026, making the year ambiguous from the source ma

### Support bot

- `what order types can i use`
  - judge: The answer claims that Polymarket prediction markets support 'Limit Order' execution 'at your specified price or better' and lists them as available 'on all markets (Perps, Spot, Predictions)', but th
