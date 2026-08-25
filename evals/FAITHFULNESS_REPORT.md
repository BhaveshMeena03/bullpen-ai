# Faithfulness — do answers stay inside their sources?

Each answer is checked against the exact material it was built from. Quotes are verified by containment, which is a fact about the output; timestamps and figures are reported with context, because rounding and unit changes are legitimate.

| surface | questions | fabricated quotes | unverified | judged unsupported |
|---|---:|---:|---:|---:|
| market search | 12 | 0 | 2 | 6 |

A non-zero **fabricated quotes** column is a release blocker: it means the tool put words in someone's mouth. The other two columns are for reading, not for gating.

### Market search

- `why does ansem think eth is going to zero`
  - [unverified] figure: $1,000 — does not appear in any retrieved window
  - judge: The answer claims Ansem predicts Ethereum will go 'to around $1,000 in the next year or so,' but the source states 'I think it's going to go to a thousand. Like in the next year or so'—this is ambiguo
- `what did the helius guy say about solana`
  - judge: The answer cites a timestamp [2:15:05] that does not exist in a ~2 hour video (the source timestamps range from 20:03 to 1:28:23), making this a fabricated citation.
- `how did that guy turn 500 dollars into 40 million`
  - [unverified] figure: $40M — does not appear in any retrieved window
  - judge: The answer claims Kimchi 'hit Trump for $40 million,' but the sources say he 'hit Trump for $40 million' in reference to profiting from a Trump-related trade, not that Trump was the counterparty or vi
- `which stocks are the robot play`
  - judge: The answer claims RFL is 'trying to create the UFC/F1 of robotics' with 'competitive robot fighting,' but the source says it's 'basically trying to create the UFC/F1 of robotics' without specifying 'c
- `mizkif giving his chat money to trade`
  - judge: The answer states Mizkif is 'giving his chat $300,000' but the source title and his own statement indicate he gave them $400,000, not $300,000.
- `selling a crypto company to kraken`
  - judge: The answer states the acquisition occurred in September 2025, but the sources indicate September 2026 (the episodes aired 2026-05-22, and he says 'Kraken acquired us in September,' referring to the pr
