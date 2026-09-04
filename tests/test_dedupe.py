import pytest


# --- one card per show ------------------------------------------------------

from app.dedupe import group_by_show, canonical_episode_ids   # noqa: E402


def _ep(eid, day, words, seconds=100):
    """An episode whose transcript is the given words, as running prose so
    the three-word runs the matcher looks at actually exist."""
    text = " ".join(words)
    return {"episode_id": eid, "published_at": day,
            "segments": [{"t": 0, "text": text}, {"t": seconds, "text": "the end"}]}


def _talk(seed, n=400):
    """Filler that reads like speech: shared small words, distinct content."""
    out = []
    for i in range(n):
        out += ["and", "then", "he", "said", f"{seed}{i}", "about", "the", "market"]
    return out


def test_a_broadcast_and_its_cut_are_one_show():
    """The 20 August evening was three rows: the 5.4h live broadcast, the
    Orangie cut, and the next day's upload. The page showed three cards."""
    body = _talk("orangie")
    live = _ep("live", "2026-08-20", body + _talk("extra", 60), seconds=19000)
    cut = _ep("cut", "2026-08-20", body[:1200], seconds=3600)
    upload = _ep("upload", "2026-08-21", body[:2000], seconds=9000)
    assert len(group_by_show([live, cut, upload])) == 1
    # The longest survives, so the full broadcast is the row that shows.
    assert canonical_episode_ids([live, cut, upload]) == {"live"}


def test_different_shows_stay_apart():
    """Two hours of crypto talk share a vocabulary; that must not merge
    them. Only the runs of words they actually share count."""
    a = _ep("a", "2026-08-20", _talk("guestone"))
    b = _ep("b", "2026-08-27", _talk("guesttwo"))
    assert len(group_by_show([a, b])) == 2


def test_the_same_guest_a_month_apart_is_two_shows():
    """The date window is what stops a returning guest merging episodes."""
    body = _talk("mizkif")
    a = _ep("a", "2026-05-07", body)
    b = _ep("b", "2026-08-07", body)
    assert len(group_by_show([a, b])) == 2


def test_grouping_survives_an_episode_with_no_transcript():
    a = _ep("a", "2026-08-20", _talk("word"))
    empty = {"episode_id": "empty", "published_at": "2026-08-20", "segments": []}
    assert len(group_by_show([a, empty])) == 2
    assert "a" in canonical_episode_ids([a, empty])


def test_the_verdict_does_not_depend_on_who_else_is_in_the_corpus():
    """The bug this replaced. Rarity was measured across every episode
    present, so the same pair grouped differently on two machines with
    different data files: production listed 20 shows while the same code
    on a laptop counted 18. A pair is now judged on the pair alone."""
    body = _talk("orangie")
    live = _ep("live", "2026-08-20", body, seconds=19000)
    cut = _ep("cut", "2026-08-20", body[:1200], seconds=3600)
    others = [_ep(f"other{i}", "2026-07-0" + str(i + 1), _talk(f"guest{i}"))
              for i in range(5)]

    alone = group_by_show([live, cut])
    crowded = group_by_show([live, cut] + others)
    merged = next(g for g in crowded
                  if {e["episode_id"] for e in g} & {"live", "cut"})
    assert len(alone) == 1
    assert {e["episode_id"] for e in merged} == {"live", "cut"}


def test_an_episode_the_data_file_has_never_seen_is_still_listed():
    """The failure this guards: summaries are written to Pinecone by the
    ingest, episodes.json ships with the image. Between an ingest and the
    next deploy the newest show is searchable and answering questions
    while being absent from the episode list. That happened."""
    known = {"a", "b"}
    canonical = {"a"}
    rows = [{"episode_id": "a"}, {"episode_id": "b"}, {"episode_id": "brand-new"}]
    shown = [r for r in rows
             if r["episode_id"] not in known or r["episode_id"] in canonical]
    assert [r["episode_id"] for r in shown] == ["a", "brand-new"]


# --- the number the show gives its own episodes -----------------------------

from app.dedupe import episode_number                          # noqa: E402


@pytest.mark.parametrize("title,expected", [
    ("LIVE W/ WILL CLEMENTE, NET NET CAPITAL: Market Bubble Ep 18", 18),
    ("We are entering a SUPERCYCLE | Market Bubble #18", 18),
    ("LIVE W/ MIZKIF: Market Bubble EP 2 - Presented by @Polymarket", 2),
    ("Why AI Is Beating Crypto Right Now | Market Bubble #2", 2),
    # Two the archive genuinely cannot parse; they fall back to the
    # transcript matcher, which is the stronger test anyway.
    ("Market Bubble: The Ansem Edition - Presented by @Polymarket", None),
    ("$100K POLYMARKET FANTASY FOOTBALL DRAFT NIGHT", None),
])
def test_episode_number(title, expected):
    assert episode_number(title) == expected


def test_the_broadcast_and_its_cut_share_a_number():
    """The pair that was listed twice on the night Ep 18 aired."""
    live = "LIVE W/ WILL CLEMENTE, NET NET CAPITAL: Market Bubble Ep 18"
    cut = "We are entering a SUPERCYCLE | Market Bubble #18"
    assert episode_number(live) == episode_number(cut)
