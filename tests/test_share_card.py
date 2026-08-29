"""The card X shows when somebody shares a search.

Two things were wrong and both only show up in somebody else's timeline,
which is the worst place to find out.

og:url was hardcoded to the empty homepage, so every shared search
canonicalised to the same URL and clicking the card landed on a blank
search box instead of the answer.

And the card was identical for every search — sharing "what did ansem and
banks say about $ansem" showed the same generic blurb as the front page.
X's crawler does not run JavaScript, so this has to be done on the way
out or not at all.

The query arrives from a URL anybody can craft and lands inside an HTML
attribute, so the escaping is not optional and is tested here rather than
assumed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

PAGE = "/demo/podcast.html"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def meta(html: str, key: str) -> str | None:
    found = re.search(
        rf'<meta (?:property|name)="{re.escape(key)}" content="([^"]*)"', html)
    return found.group(1) if found else None


def test_a_shared_search_is_titled_with_the_question(client):
    r = client.get(PAGE, params={"q": "what did ansem say about hyperliquid"})
    assert r.status_code == 200
    for key in ("og:title", "twitter:title"):
        assert "what did ansem say about hyperliquid" in meta(r.text, key)


def test_the_front_page_keeps_its_own_title(client):
    r = client.get(PAGE)
    assert "Market Bubble Search" in meta(r.text, "og:title")


def test_the_canonical_url_is_the_url_that_was_shared(client):
    """Hardcoded to the homepage, every search link collapsed to it and
    the card sent people to an empty box."""
    r = client.get(PAGE, params={"q": "what did banks say about streaming"})
    url = meta(r.text, "og:url")
    assert "q=" in url and "streaming" in url
    assert "v=2" not in url


def test_a_crafted_query_cannot_break_out_of_the_attribute(client):
    r = client.get(PAGE, params={"q": '"><script>alert(1)</script>'})
    assert "<script>alert(1)" not in r.text
    title = meta(r.text, "og:title")
    assert "&lt;script&gt;" in title or "&lt;" in title
    # The quote that would end the attribute early must be encoded.
    assert '"><script' not in title


def test_a_long_query_is_trimmed_rather_than_dumped(client):
    r = client.get(PAGE, params={"q": "A" * 500})
    assert len(meta(r.text, "og:title")) < 200


def test_the_card_image_is_still_declared(client):
    """The title is the change; losing the image would trade a generic
    card for a bare link, which is worse than what it replaced."""
    r = client.get(PAGE, params={"q": "anything"})
    assert meta(r.text, "og:image", ).endswith(".png")
    assert meta(r.text, "twitter:card") == "summary_large_image"


def test_the_page_itself_still_loads(client):
    """It is served by a route now rather than the static mount, so the
    page has to arrive intact — not only its meta tags."""
    r = client.get(PAGE)
    assert "<title>" in r.text
    assert 'id="q"' in r.text
    assert len(r.text) > 20000
