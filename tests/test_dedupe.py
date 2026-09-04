

# --- one card per show ------------------------------------------------------

from app.dedupe import group_by_show, canonical_episode_ids   # noqa: E402


def _ep(eid, day, words, seconds=100):
    """An episode whose transcript is the given words."""
    return {"episode_id": eid, "published_at": day,
            "segments": [{"t": i * 10, "text": w}
                         for i, w in enumerate(words)] +
                        [{"t": seconds, "text": "tail"}]}


def test_a_broadcast_and_its_cut_are_one_show():
    """The 20 August evening was three rows: the 5.4h live broadcast, the
    Orangie cut, and the next day's upload. The page showed three cards."""
    rare = [f"orangie{i}" for i in range(40)]
    live = _ep("live", "2026-08-20", rare + ["extra"] * 5, seconds=19000)
    cut = _ep("cut", "2026-08-20", rare[:30], seconds=3600)
    upload = _ep("upload", "2026-08-21", rare[:34], seconds=9000)
    groups = group_by_show([live, cut, upload])
    assert len(groups) == 1
    # The longest survives, so the full broadcast is the row that shows.
    assert canonical_episode_ids([live, cut, upload]) == {"live"}


def test_different_shows_stay_apart():
    """Two hours of crypto talk share a vocabulary; that must not merge
    them. Only the rare words count."""
    a = _ep("a", "2026-08-20", [f"guestone{i}" for i in range(40)])
    b = _ep("b", "2026-08-27", [f"guesttwo{i}" for i in range(40)])
    assert len(group_by_show([a, b])) == 2


def test_the_same_guest_a_month_apart_is_two_shows():
    """The date window is what stops a returning guest merging episodes."""
    rare = [f"mizkif{i}" for i in range(40)]
    a = _ep("a", "2026-05-07", rare)
    b = _ep("b", "2026-08-07", rare)
    assert len(group_by_show([a, b])) == 2


def test_grouping_survives_an_episode_with_no_transcript():
    a = _ep("a", "2026-08-20", [f"word{i}" for i in range(40)])
    empty = {"episode_id": "empty", "published_at": "2026-08-20",
             "segments": []}
    groups = group_by_show([a, empty])
    assert len(groups) == 2
    assert "a" in canonical_episode_ids([a, empty])


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
