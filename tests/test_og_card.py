"""The share card, drawn around the question that was asked.

The card used to be a static PNG with an example question printed into
its mock search box. Sharing a search then showed the reader two
different questions at once: the one baked into the image, and the real
one X overlays from og:title. It looked like a mistake because it was.

The question arrives from a URL anybody can craft, so most of what is
guarded here is the drawing surviving input nobody would type.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app import og_card  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestShipsWithWhatItNeeds:
    def test_the_fonts_are_in_the_repo(self):
        """python:3.12-slim has no fonts at all, not even DejaVu, so
        these have to travel with the image or every card is a 500."""
        fonts = ROOT / "demo" / "fonts"
        assert (fonts / "Inter-Variable.ttf").exists()
        assert (fonts / "InstrumentSerif-Regular.ttf").exists()

    def test_the_image_copies_them(self):
        assert "COPY demo ./demo" in (ROOT / "Dockerfile").read_text()

    def test_pillow_is_a_declared_dependency(self):
        """It was not, and the route would have ImportError'd on the
        first share while working locally."""
        assert "pillow" in (ROOT / "requirements.txt").read_text().lower()


class TestDrawing:
    def test_a_question_produces_a_png_of_the_right_size(self):
        img = Image.open(io.BytesIO(og_card.render("what did ansem say")))
        assert img.format == "PNG"
        assert img.size == (1200, 630)

    def test_the_bare_card_still_draws(self):
        img = Image.open(io.BytesIO(og_card.render()))
        assert img.size == (1200, 630)

    def test_a_question_changes_the_pixels(self):
        """If it did not, the whole point of the route is missing."""
        assert og_card.render("a question") != og_card.render()

    def test_two_questions_differ(self):
        assert og_card.render("solana") != og_card.render("ethereum")

    @pytest.mark.parametrize("hostile", [
        "A" * 600,
        "supercalifragilistic" * 40,          # one unbreakable token
        '"><script>alert(1)</script>',
        "什么是超流动性协议的市值",              # no glyphs in either face
        "",
        "   ",
        "🙂" * 50,
    ])
    def test_it_survives_whatever_arrives_in_the_url(self, hostile):
        img = Image.open(io.BytesIO(og_card.render(hostile)))
        assert img.size == (1200, 630)


class TestRoute:
    def test_it_serves_a_png(self, client):
        r = client.get("/og/search.png", params={"q": "what did banks say"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_it_is_cached_and_revalidates(self, client):
        """A crawler fetches each card once. Without this every share of
        the same link redraws it."""
        r = client.get("/og/search.png", params={"q": "cached please"})
        assert "max-age" in r.headers["cache-control"]
        again = client.get("/og/search.png", params={"q": "cached please"},
                           headers={"if-none-match": r.headers["etag"]})
        assert again.status_code == 304

    def test_the_etag_follows_the_question(self):
        assert og_card.etag("one") != og_card.etag("two")
        assert og_card.etag("one") == og_card.etag("one")


class TestThePagePointsAtIt:
    def test_a_search_gets_its_own_card(self, client):
        r = client.get("/demo/podcast.html",
                       params={"q": "who sold all his eth"})
        assert "/og/search.png?q=who+sold+all+his+eth" in r.text

    def test_the_front_page_keeps_the_static_one(self, client):
        """Nothing to draw a question around, and the static card is
        already right for it."""
        r = client.get("/demo/podcast.html")
        assert "og-broadcast.png" in r.text
        assert "/og/search.png" not in r.text

    def test_both_image_tags_move_together(self, client):
        """twitter:image left behind would show the old card on X, which
        is the only place this matters."""
        r = client.get("/demo/podcast.html", params={"q": "hyperliquid"})
        assert r.text.count("/og/search.png") == 2
