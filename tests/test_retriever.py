"""Concierge retriever reranking — brings it to parity with the podcast
index. The retriever must widen its candidate fetch when reranking is on,
reorder by the reranker's verdict, truncate to top_k, and — critically —
fall back to vector order if reranking fails (never break search)."""

import asyncio
import types

import pytest

import app.retriever as retriever_mod
from app.retriever import Retriever


def _match(id_: str, score: float, text: str):
    return types.SimpleNamespace(
        id=id_, score=score,
        metadata={"text": text, "source_type": "docs", "title": id_},
    )


class _FakeIndex:
    def __init__(self, matches, capture):
        self._matches = matches
        self._capture = capture

    def query(self, vector, top_k, filter, include_metadata):  # noqa: A002
        self._capture["fetch_k"] = top_k
        return types.SimpleNamespace(matches=self._matches)


@pytest.fixture
def retriever(monkeypatch):
    async def fake_embed(client, text, *, model, dimension):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(retriever_mod, "embed_query", fake_embed)
    return Retriever()


def test_widens_fetch_reranks_and_truncates(retriever, monkeypatch):
    capture = {}
    matches = [_match(f"c{i}", 0.9 - i * 0.01, f"text {i}") for i in range(6)]
    retriever._index = _FakeIndex(matches, capture)

    async def fake_rerank(client, query, docs, *, top_k, model):
        return list(reversed(range(len(docs))))[:top_k]  # best-first = reversed

    monkeypatch.setattr(retriever_mod, "rerank_order", fake_rerank)

    out = asyncio.run(retriever.search("q", top_k=3))

    assert capture["fetch_k"] == retriever._settings.rerank_candidates  # widened
    assert len(out) == 3                       # truncated to top_k
    assert out[0].id == "c5"                   # reranker put the last chunk first


def test_rerank_failure_falls_back_to_vector_order(retriever, monkeypatch):
    matches = [_match(f"c{i}", 0.9 - i * 0.01, f"text {i}") for i in range(6)]
    retriever._index = _FakeIndex(matches, {})

    async def fake_rerank(*a, **k):
        return None  # rerank down (rate limit / API change)

    monkeypatch.setattr(retriever_mod, "rerank_order", fake_rerank)

    out = asyncio.run(retriever.search("q", top_k=3))
    assert [c.id for c in out] == ["c0", "c1", "c2"]  # original order preserved
