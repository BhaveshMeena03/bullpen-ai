"""AssetStore — per-episode hits in Pinecone, so the DEPLOYED dashboard
updates when the weekly sync runs, without a redeploy.

Pinecone is stubbed: these cover the parts that can actually be wrong —
the metadata size cap, the round-trip through the JSON blob, and degrading
instead of exploding on a corrupt blob.
"""

import asyncio
import json
import types

import pytest

from app.assets_store import MAX_METADATA_CHARS, AssetStore


class _FakeIndex:
    def __init__(self, vectors=None):
        self.upserted = []
        self._vectors = vectors or {}

    def upsert(self, vectors, namespace):
        self.upserted.append((vectors, namespace))
        for v in vectors:
            self._vectors[v["id"]] = types.SimpleNamespace(metadata=v["metadata"])

    def list(self, namespace):
        yield types.SimpleNamespace(
            vectors=[types.SimpleNamespace(id=k) for k in self._vectors]
        )

    def delete(self, ids, namespace):
        for i in ids:
            self._vectors.pop(i, None)

    def fetch(self, ids, namespace):
        return types.SimpleNamespace(
            vectors={i: self._vectors[i] for i in ids if i in self._vectors}
        )


@pytest.fixture
def store():
    s = AssetStore()
    s._index = _FakeIndex()
    return s


def _hits(n, note="x"):
    return [{"symbol": "SOL", "name": "Solana", "kind": "analysis",
             "start_seconds": float(i), "note": note, "confidence": "high",
             "episode_id": "e1", "episode_title": "Ep", "url": "https://y/1"}
            for i in range(n)]


def test_store_roundtrips_hits(store):
    n = asyncio.run(store.store("e1", "Ep One", _hits(3)))
    assert n == 3
    out = asyncio.run(store.all_hits())
    assert len(out) == 3
    assert out[0]["symbol"] == "SOL"


def test_oversized_episode_is_chunked_not_truncated(store):
    """Regression: one record per episode silently dropped the tail of 8 of
    15 real episodes, which serialize to 30-48KB against a 40KB cap."""
    big = _hits(400, note="y" * 200)
    assert len(json.dumps(big)) > MAX_METADATA_CHARS

    n = asyncio.run(store.store("e1", "Ep", big))
    assert n == 400, "every hit must be stored, not trimmed"

    vectors, _ = store._index.upserted[0]
    assert len(vectors) > 1, "should have split across records"
    for v in vectors:
        assert len(v["metadata"]["hits"]) <= MAX_METADATA_CHARS

    # And it must all read back.
    assert len(asyncio.run(store.all_hits())) == 400


def test_restore_with_fewer_hits_removes_stale_chunks(store):
    asyncio.run(store.store("e1", "Ep", _hits(400, note="y" * 200)))
    many = len(store._index._vectors)
    asyncio.run(store.store("e1", "Ep", _hits(2)))
    assert len(store._index._vectors) < many, "stale chunks must be deleted"
    assert len(asyncio.run(store.all_hits())) == 2


def test_corrupt_blob_is_skipped_not_raised(store):
    store._index._vectors["assets-bad"] = types.SimpleNamespace(
        metadata={"hits": "{not json"}
    )
    asyncio.run(store.store("e1", "Ep", _hits(2)))
    out = asyncio.run(store.all_hits())
    assert len(out) == 2  # the good episode survives the bad blob


def test_writes_to_assets_namespace_with_stable_id(store):
    asyncio.run(store.store("abc123", "Ep", _hits(1)))
    vectors, namespace = store._index.upserted[0]
    assert namespace == "assets"
    assert vectors[0]["id"] == "assets-abc123-0"  # stable => re-runs overwrite


def test_exists_reflects_store(store):
    assert asyncio.run(store.exists("e1")) is False
    asyncio.run(store.store("e1", "Ep", _hits(1)))
    assert asyncio.run(store.exists("e1")) is True


class _HangingIndex(_FakeIndex):
    """upsert() blocks forever, mimicking a dead-but-ESTABLISHED socket."""

    def upsert(self, vectors, namespace):
        # Long enough to blow past the test's 0.2s timeout, short enough that
        # the orphaned worker thread (wait_for can't kill it) doesn't wedge
        # interpreter shutdown.
        import time
        time.sleep(5)


def test_hung_write_times_out_instead_of_blocking(store):
    # The real incident: a write hung 2.5h on a half-open socket because the
    # Pinecone client has no read timeout. The wait_for guard must convert that
    # into a prompt failure the caller's idempotent retry can recover from.
    store._index = _HangingIndex()
    store._settings.pinecone_write_timeout_seconds = 0.2
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        asyncio.run(store.store("e1", "Ep", _hits(1)))
