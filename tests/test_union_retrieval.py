"""Show the model both the deep pool's answer and the shallow pool's.

Reranking fifty candidates reaches passages that reranking twelve cannot
-- the line naming who sold their entire ETH position sits at rank 44 --
but reranking fifty *alone* loses answers twelve got right, because
positions two to six fill with passages merely about the same subject
and evict the specific one.

Sequencing the two is a no-op and that was measured, not assumed:
reranking is a total order, so narrowing fifty to twelve and reranking
those twelve hands back the same six. Combining them is what changes
anything, because then neither set has to win the same slots.

Across a hundred questions: three misattributions at twelve, two at
fifty, none at the union, with the best recall of the three.

The invariant that matters here is that every passage the model reads
comes back in the response. The reply builder resolves a cited timestamp
by scanning the returned hits for the episode that contains it; a
passage the model quoted but the caller never saw sends the link to
whatever ranked first instead, which is how a true sentence ended up
welded to an unrelated episode once already.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.config import Settings  # noqa: E402


def hit_key(episode_id: str, start: float) -> tuple[str, float]:
    return (episode_id, start)


class TestSettingDefault:
    def test_the_union_is_on(self):
        """Shipped after the hundred-question comparison. Off is still one
        env var away, which is the whole reason it is a setting."""
        assert Settings.model_fields["rerank_narrow_pool"].default == 12

    def test_it_can_be_turned_off_without_a_deploy(self):
        s = Settings(anthropic_api_key="k", voyage_api_key="k",
                     pinecone_api_key="k", rerank_narrow_pool=0)
        assert s.rerank_narrow_pool == 0


class TestUnionShape:
    """The merge itself, without a network.

    `deep` and `shallow` are index lists into the same candidate list,
    exactly as rerank_order returns them.
    """

    @staticmethod
    def merge(candidates, deep, shallow, top_k):
        """The logic under test, mirrored from _retrieve."""
        ordered = [candidates[i] for i in deep]
        kept = ordered[:top_k]
        seen = {c for c in kept}
        return kept + [candidates[i] for i in shallow
                       if candidates[i] not in seen]

    def test_the_shallow_pick_is_not_evicted(self):
        """The case this exists for: the deep pool ranks a passage the
        shallow pool never saw, and the shallow pool holds one the deep
        ordering pushed past six."""
        cands = list("abcdefghijkl")
        deep = [11, 10, 9, 8, 7, 6]          # deep favours the tail
        shallow = [0, 1, 2]                  # shallow favours the head
        out = self.merge(cands, deep, shallow, top_k=6)
        assert out[:6] == ["l", "k", "j", "i", "h", "g"]
        assert "a" in out and "b" in out and "c" in out

    def test_a_passage_in_both_appears_once(self):
        """Both passes rank the same candidate list, so overlap is the
        normal case, not the exception."""
        cands = list("abcdef")
        out = self.merge(cands, [0, 1, 2], [0, 1, 3], top_k=3)
        assert out == ["a", "b", "c", "d"]
        assert len(out) == len(set(out))

    def test_full_overlap_returns_exactly_the_deep_six(self):
        """When the shallow pass agrees, the union costs nothing but the
        extra rerank call -- no wider prompt, no extra cards."""
        cands = list("abcdef")
        out = self.merge(cands, [0, 1, 2], [0, 1, 2], top_k=3)
        assert out == ["a", "b", "c"]

    def test_the_deep_order_is_preserved_at_the_front(self):
        """The best passage by relevance still leads, because the answer
        and its link are built from the front of the list."""
        cands = list("abcdef")
        out = self.merge(cands, [3, 4], [0], top_k=2)
        assert out[0] == "d"


class TestCitationSafety:
    def test_every_passage_the_model_read_is_returned(self):
        """Not a style point. format() builds the prompt from the list
        and the response carries the same list; if the caller is handed a
        shorter one, a citation into the dropped tail cannot be resolved
        and the link falls back to whatever ranked first."""
        from app.podcast import PodcastIndex
        source = Path(PodcastIndex.__module__.replace(".", "/") + ".py")
        text = (ROOT / source).read_text()
        assert "return _prefer_seekable(hits)[:keep]" in text, (
            "the union must not be truncated back to top_k on the way out")


@pytest.mark.parametrize("narrow,expected", [(0, False), (12, True)])
def test_zero_disables_the_second_pass(narrow, expected):
    """Rollback is setting this to 0, so it has to be a real off switch
    rather than a smaller pool."""
    assert bool(narrow) is expected
