# Faithfulness — do answers stay inside their sources?

Each answer is checked against the exact material it was built from. Quotes are verified by containment, which is a fact about the output; timestamps and figures are reported with context, because rounding and unit changes are legitimate.

| surface | questions | fabricated quotes | unverified | judged unsupported |
|---|---:|---:|---:|---:|
| market search | 12 | 0 | 1 | 4 |

A non-zero **fabricated quotes** column is a release blocker: it means the tool put words in someone's mouth. The other two columns are for reading, not for gating.

### Market search

- `what did the helius guy say about solana`
  - judge: The answer cites a timestamp [2:15:04] that does not exist in the provided sources, which are only up to 1:28:23, making it impossible to verify the quoted claim about Solana being 'the most performan
- `the episode about someone suing the sec`
  - judge: ungradeable judge output: ```json
{
  "verdict": "FAIL",
  "why": "The answer states the court 'awarded da
- `which stocks are the robot play`
  - judge: The answer claims SpaceX IPO'd at $135 per share, but the sources only state it was 'priced at 135 a share' during the IPO process—not that it actually IPO'd at that price. The sources indicate it 'wa
- `mizkif giving his chat money to trade`
  - [unverified] figure: $100,000 — does not appear in any retrieved window
  - judge: The answer attributes statements to 'Ansem' when the sources show these statements were made by someone else (the speaker on 'Kick' who is giving their chat $300,000), not by Ansem, who is a different
