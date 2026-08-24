"""Fetch the official ClawPump documentation into concierge knowledge-base
documents.

Same job as fetch_bullpen_docs.py, one structural difference: Bullpen runs
GitBook, where every page has a `.md` twin, so that fetcher just downloads
Markdown. ClawPump is a Next.js site with no Markdown twin at all — the
prose only exists inside rendered HTML.

That matters more than it sounds. The docs chunker in app/ingest.py splits
on Markdown headings, so flattening these pages to plain text would hand it
one undifferentiated wall per page, and a question about fees would retrieve
a window that happens to mention fees in passing rather than the section
about them. So this converts HTML to Markdown and *keeps the heading
levels*, which is what makes the chunk boundaries land on real section
boundaries.

Two other things are done for retrieval quality rather than tidiness:

  boilerplate  The nav ("Tokens Analytics Leaderboard Stories...") and the
               footer repeat on all 40 pages. Left in, every chunk shares
               that text, every page looks slightly like every other page,
               and the embeddings lose separation. They are stripped.
  headings     Each chunk inherits the heading path above it, so a chunk
               under "## Fees > ### Creator share" carries those words even
               when the sentence itself only says "65%".

Re-runnable: run it again to refresh the KB when the docs change.

    python scripts/fetch_clawpump_docs.py
"""

from __future__ import annotations

import json
import re
import sys
from html import unescape
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import IngestDocument  # noqa: E402

BASE = "https://clawpump.tech"
SITEMAP = f"{BASE}/sitemap.xml"
OUT = Path(__file__).resolve().parent.parent / "data" / "clawpump_docs.json"

# The sitemap carries ~200 token detail pages and 13 translations of the
# hackathon page. Neither is documentation: token pages are live market data
# that is stale the moment it is embedded, and a Turkish copy of a page we
# already have in English adds nothing but duplicate neighbours for every
# nearby query.
_SKIP_PREFIX = ("/tokens/",)
_SKIP_LOCALE = re.compile(r"^/(es|pt|zh|ja|ko|tr|ru|uk|ar|fr|de|sw|ha|yo)/")
# Pages whose content is a live listing rather than prose. Embedding a
# leaderboard teaches the retriever nothing and dates instantly.
_SKIP_EXACT = {"/tokens", "/leaderboard", "/analytics", "/marketplace",
               "/status", "/sitemap.xml"}

# Fallback list, used only when the sitemap cannot be fetched, so a network
# blip refreshes a slightly stale KB instead of silently emptying it.
FALLBACK_PAGES = [
    "/docs", "/mcp", "/guide", "/developers", "/docs-x", "/why-deploy",
    "/claw-agent", "/ansemhack", "/stories", "/blog", "/experiments",
]

_SCRIPT = re.compile(r"<(script|style|noscript|svg)[\s\S]*?</\1>", re.I)
_COMMENT = re.compile(r"<!--[\s\S]*?-->")
_NAV = re.compile(r"<(nav|header|footer)\b[\s\S]*?</\1>", re.I)
_TAG = re.compile(r"<[^>]+>")
_MULTI_BLANK = re.compile(r"\n{3,}")
_HEADING = re.compile(r"<h([1-6])\b[^>]*>([\s\S]*?)</h\1>", re.I)
_LIST_ITEM = re.compile(r"<li\b[^>]*>([\s\S]*?)</li>", re.I)
_PARA = re.compile(r"<(p|div)\b[^>]*>([\s\S]*?)</\1>", re.I)
_CODE = re.compile(r"<(code|pre)\b[^>]*>([\s\S]*?)</\1>", re.I)


def _inline(html: str) -> str:
    """Tags stripped, entities decoded, whitespace collapsed."""
    return re.sub(r"\s+", " ", unescape(_TAG.sub(" ", html))).strip()


def to_markdown(html: str) -> str:
    """Rendered page -> Markdown, keeping headings and list structure.

    Deliberately not a general HTML converter. It walks the document once,
    emitting a line for each block-level element it recognises, because the
    only structure the downstream chunker reads is `#` heading depth.
    """
    html = _COMMENT.sub(" ", html)
    html = _SCRIPT.sub(" ", html)
    html = _NAV.sub(" ", html)

    out: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(
        r"<h([1-6])\b[^>]*>([\s\S]*?)</h\1>"
        r"|<li\b[^>]*>([\s\S]*?)</li>"
        r"|<(?:p)\b[^>]*>([\s\S]*?)</p>"
        r"|<pre\b[^>]*>([\s\S]*?)</pre>",
        re.I,
    )
    for m in pattern.finditer(html):
        level, head, item, para, code = m.groups()
        if head is not None:
            text = _inline(head)
            if text:
                out.append(f"\n{'#' * int(level)} {text}\n")
        elif item is not None:
            text = _inline(item)
            if text and text not in seen:
                seen.add(text)
                out.append(f"- {text}")
        elif para is not None:
            text = _inline(para)
            # Nested <p> inside <div> yields the same prose twice; a repeated
            # sentence would be embedded twice and crowd out its neighbours.
            if text and len(text) > 2 and text not in seen:
                seen.add(text)
                out.append(text)
        elif code is not None:
            text = _inline(code)
            if text:
                out.append(f"```\n{text}\n```")
    return _MULTI_BLANK.sub("\n\n", "\n".join(out)).strip()


def title_of(html: str, slug: str) -> str:
    m = re.search(r"<title\b[^>]*>([\s\S]*?)</title>", html, re.I)
    if m:
        raw = _inline(m.group(1))
        # Their titles are "Documentation — MCP Tools, Skills & API | ClawPump".
        # The suffix is on every page and carries no signal.
        raw = re.split(r"\s*\|\s*ClawPump\s*$", raw)[0].strip()
        if raw:
            return raw
    return slug.strip("/").replace("/", ": ").replace("-", " ").title()


def _keep(path: str) -> bool:
    return not (path in _SKIP_EXACT
                or path.startswith(_SKIP_PREFIX)
                or bool(_SKIP_LOCALE.match(path)))


def discover_pages(client: httpx.Client) -> list[str]:
    """Documentation paths, from the sitemap AND llms.txt together.

    Neither source is complete on its own. The sitemap omits /docs-x, the
    page documenting the tweet-to-launch commands — which is precisely the
    kind of thing someone asks a support bot about, so losing it would be a
    hole in the answers rather than a few missing characters. llms.txt is
    the index the site publishes for machines and lists it, but it in turn
    omits individual blog and story pages the sitemap has.

    Taking the union means a page has to be missing from both to be missed,
    and adding a source can only ever widen coverage.
    """
    paths: set[str] = set()

    try:
        resp = client.get(SITEMAP)
        resp.raise_for_status()
        paths |= {loc.strip().removeprefix(BASE) or "/"
                  for loc in re.findall(r"<loc>([^<]+)</loc>", resp.text)}
    except httpx.HTTPError as exc:
        print(f"  sitemap unavailable ({exc})")

    try:
        resp = client.get(f"{BASE}/llms.txt")
        resp.raise_for_status()
        found = {u.removeprefix(BASE)
                 for u in re.findall(r"https://clawpump\.tech(?:/[\w./-]*)?",
                                     resp.text)}
        new = {p or "/" for p in found} - paths
        if new:
            print(f"  llms.txt adds {len(new)} page(s) the sitemap omits: "
                  f"{', '.join(sorted(new)[:6])}")
        paths |= {p or "/" for p in found}
    except httpx.HTTPError as exc:
        print(f"  llms.txt unavailable ({exc})")

    kept = sorted(p for p in paths if _keep(p))

    # Third pass: follow the links on the pages we already have.
    #
    # This exists because of a real miss on the Bullpen side of this repo.
    # That fetcher started from a hand-written page list, which silently went
    # stale and left the support bot with nothing on stop-limit orders, TWAP
    # or time-in-force — most of what people actually asked it. Switching to
    # the sitemap fixed that, and the lesson generalises: any single index of
    # a site is something a human maintains, and humans forget pages.
    #
    # Here the sitemap and llms.txt between them both omit /terms and
    # /privacy, which are linked from the footer of every page. A support bot
    # that cannot answer "what are the terms" has the same hole, just smaller.
    kept = sorted(set(kept) | _crawl_links(client, kept))
    if not kept:
        print("  discovery found nothing — falling back to curated list")
        return list(FALLBACK_PAGES)
    return kept


def _crawl_links(client: httpx.Client, seeds: list[str]) -> set[str]:
    """Internal links found on the seed pages that no index listed.

    One hop, not a full crawl: the seeds already include every section entry
    point, so anything reachable and worth having is one link away. A page is
    kept only if it survives the same filters and actually renders prose —
    which is what excludes the dashboard, whose routes are auth-gated shells
    that return a full HTML document and no content at all.
    """
    seen = set(seeds)
    found: set[str] = set()
    for path in seeds:
        try:
            html = client.get(f"{BASE}{path}").text
        except httpx.HTTPError:
            continue
        for href in re.findall(r'href="(/[^"#?]*)"', html):
            candidate = href.rstrip("/") or "/"
            if candidate in seen or candidate in found or not _keep(candidate):
                continue
            if re.search(r"\.(png|jpe?g|svg|ico|webp|xml|txt|json)$", candidate, re.I):
                continue
            found.add(candidate)

    real = set()
    for candidate in sorted(found):
        try:
            resp = client.get(f"{BASE}{candidate}")
            resp.raise_for_status()
        except httpx.HTTPError:
            continue
        if len(to_markdown(resp.text)) >= 400:
            real.add(candidate)
    if real:
        print(f"  link crawl adds {len(real)} unlisted page(s): "
              f"{', '.join(sorted(real))}")
    return real


_CHANNEL_RE = re.compile(
    r'href="(https://(?:discord\.gg|discord\.com/invite|t\.me|github\.com/[Cc]lawpump'
    r'|x\.com/clawpumptech)[^"]*)"')


def report_channels(client: httpx.Client, paths: list[str]) -> None:
    """Print the official off-site channels linked from ClawPump's pages.

    These are NOT ingested, on purpose. Contact details are the one thing a
    support bot must never repeat from scraped text — a poisoned Discord
    invite in a retrieved chunk reads exactly like a real one, and the whole
    value of the pinned <official_channels> list in the prompt is that a
    human reviewed every entry.

    But "not ingested" turned into "invisible". The link crawl only follows
    internal paths, so the Discord invite in the site footer was never seen
    by anything, and the bot answered "I don't have that" to the single most
    common support question there is. Printing them on every refresh puts
    new or changed channels in front of whoever runs this, who can then
    decide to pin them. The review stays manual; only the noticing is
    automated.
    """
    found: set[str] = set()
    for path in paths:
        try:
            found |= set(_CHANNEL_RE.findall(client.get(f"{BASE}{path}").text))
        except httpx.HTTPError:
            continue
    if found:
        print("\n  official channels linked from the site — pin any that are")
        print("  missing from <official_channels> in app/clawpump.py:")
        for url in sorted(found):
            print(f"    {url}")


def main() -> None:
    docs: list[dict] = []
    with httpx.Client(timeout=40, follow_redirects=True,
                      headers={"user-agent": "clawpump-concierge/1.0"}) as client:
        pages = discover_pages(client)
        print(f"  {len(pages)} pages to fetch\n")
        for path in pages:
            try:
                resp = client.get(f"{BASE}{path}")
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  SKIP {path}: {exc}")
                continue
            body = to_markdown(resp.text)
            if len(body) < 200:
                print(f"  SKIP {path}: too short after cleaning ({len(body)})")
                continue
            title = title_of(resp.text, path)
            doc = IngestDocument(
                source_type="docs",
                source_id=f"cp-{path.strip('/').replace('/', '-') or 'home'}",
                title=title,
                text=body,
                url=f"{BASE}{path}",
            )
            docs.append(doc.model_dump())
            print(f"  ok  {title[:46]:48s} {len(body):6d} chars")

        report_channels(client, pages)

    OUT.write_text(json.dumps(docs, indent=2, ensure_ascii=False))
    total = sum(len(d["text"]) for d in docs)
    print(f"\n{len(docs)}/{len(pages)} pages, {total:,} chars -> {OUT.name}")


if __name__ == "__main__":
    main()
