"""Fetch the official Bullpen documentation and turn it into concierge
knowledge-base documents.

The support agent should answer operational questions (funding, wallets,
order types, fees, execution) from Bullpen's *real* docs — not hand-written
placeholders. This pulls the Markdown source of each page (every docs URL
has a `.md` twin), strips GitBook authoring syntax, and writes
`data/bullpen_docs.json` in the IngestDocument shape.

Re-runnable: run it again to refresh the KB when the docs change.

    python scripts/fetch_bullpen_docs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import IngestDocument  # noqa: E402

BASE = "https://docs.bullpen.fi"
OUT = Path(__file__).resolve().parent.parent / "data" / "bullpen_docs.json"

# Curated for what users actually ask support about. Deep per-order-type
# sub-pages are covered by their parent "order-types" page, so they're left
# out to keep the KB focused.
PAGES = [
    "about-bullpen/what-is-bullpen",
    "about-bullpen/platform-architecture",
    "getting-started/creating-an-account",
    "getting-started/funding-your-wallet",
    "getting-started/wallet-management",
    "getting-started/account-types",
    "prediction-markets/overview",
    "prediction-markets/polymarket-integration",
    "prediction-markets/order-types",
    "prediction-markets/positions-and-settlement",
    "prediction-markets/wallets-and-collateral",
    "smart-money/overview",
    "smart-money/copy-trading",
    "smart-money/smart-money-discovery",
    "smart-money/convergence-signals",
    "smart-money/smart-money-alerts",
    "trading/overview",
    "trading/perpetuals",
    "trading/spot-trading",
    "trading/order-types",
    "trading/margin-and-leverage",
    "execution/trade-routing",
    "execution/hyperliquid-integration",
    "execution/jupiter-ultra-integration",
    "execution/unit-integration",
    "features/tradingview",
    "features/runners-memescope",
    "features/smart-wallet-tracking",
    "features/transfers",
    "features/notifications",
    "features/social-leaderboard-and-competitions",
    "features/mobile-desktop-app",
    "rewards/points",
    "rewards/referral-program-1",
    "rewards/affiliate-program",
    "rewards/vip-program-coming-soon",
    "rewards/status-match",
    "support/faqs",
    "support/glossary",
    "support/bug-bounty-program",
]

# GitBook authoring constructs to unwrap or drop.
_BLOCK_TAG = re.compile(r"\{%[^%]*%\}")          # {% hint %}, {% endhint %}, ...
_EMBED_LINE = re.compile(r"^\s*\{%\s*embed.*%\}\s*$", re.MULTILINE)
_FIGURE = re.compile(r"<figure>.*?</figure>", re.DOTALL)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LEAD_NOTE = re.compile(
    r"^>\s*For the complete documentation index.*?\n", re.MULTILINE
)
_MULTI_BLANK = re.compile(r"\n{3,}")


def clean(markdown: str) -> str:
    text = _LEAD_NOTE.sub("", markdown)
    text = _EMBED_LINE.sub("", text)
    text = _FIGURE.sub("", text)
    text = _IMAGE.sub("", text)
    # Unwrap remaining GitBook tags (hint/tabs/content-ref) but keep their
    # inner prose, which is real documentation content.
    text = _BLOCK_TAG.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def title_of(markdown: str, slug: str) -> str:
    section = slug.split("/", 1)[0].replace("-", " ").title()
    h1 = next(
        (ln[2:].strip() for ln in markdown.splitlines() if ln.startswith("# ")),
        None,
    )
    if not h1:
        return slug.rsplit("/", 1)[-1].replace("-", " ").title()
    # Several sections each have a page literally titled "Overview" — prefix
    # with the section so a citation reads "Trading: Overview", not "Overview".
    if h1.lower() in {"overview"}:
        return f"{section}: {h1}"
    return h1


def main() -> None:
    docs: list[dict] = []
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for slug in PAGES:
            url = f"{BASE}/{slug}.md"
            try:
                resp = client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  SKIP {slug}: {exc}")
                continue
            body = clean(resp.text)
            if len(body) < 80:
                print(f"  SKIP {slug}: too short after cleaning")
                continue
            title = title_of(resp.text, slug)
            doc = IngestDocument(
                source_type="docs",
                source_id=f"bp-{slug.replace('/', '-')}",
                title=title,
                text=body,
            )
            # Validate + normalize through the schema before writing.
            docs.append(doc.model_dump())
            print(f"  ok  {title[:48]:50s} {len(body):5d} chars")

    OUT.write_text(json.dumps(docs, indent=2, ensure_ascii=False))
    print(f"\n{len(docs)}/{len(PAGES)} pages -> {OUT.name}")


if __name__ == "__main__":
    main()
