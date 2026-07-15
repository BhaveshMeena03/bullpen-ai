"""Ingestion pipeline: chunking strategies and metadata tagging."""

from app.config import get_settings
from app.ingest import chunk_markdown, chunk_transcript, chunk_tweets
from app.schemas import IngestDocument, SourceType

MAX = 400
OVERLAP = 60


def doc(source_type: SourceType, text: str) -> IngestDocument:
    return IngestDocument(source_type=source_type, source_id="t1", text=text)


class TestMarkdown:
    def test_splits_on_headings(self):
        text = "# A\n" + "alpha " * 50 + "\n## B\n" + "beta " * 50
        chunks = chunk_markdown(doc(SourceType.DOCS, text), MAX, OVERLAP)
        assert len(chunks) >= 2
        assert chunks[0].text.startswith("# A")
        assert any("## B" in c.text for c in chunks)

    def test_windows_respect_size_bound(self):
        text = "\n".join(f"# H{i}\n" + "word " * 100 for i in range(8))
        chunks = chunk_markdown(doc(SourceType.DOCS, text), MAX, OVERLAP)
        # Hard bound: window + carried overlap tail + separator.
        assert all(len(c.text) <= MAX + OVERLAP + 1 for c in chunks)

    def test_oversized_single_section_is_split(self):
        text = "# Big\n" + "x" * (MAX * 3)
        chunks = chunk_markdown(doc(SourceType.DOCS, text), MAX, OVERLAP)
        assert len(chunks) >= 3
        assert all(len(c.text) <= MAX + OVERLAP + 1 for c in chunks)

    def test_no_content_lost(self):
        text = "# A\nhello unique_marker_123\n# B\nworld unique_marker_456"
        chunks = chunk_markdown(doc(SourceType.DOCS, text), MAX, OVERLAP)
        joined = " ".join(c.text for c in chunks)
        assert "unique_marker_123" in joined
        assert "unique_marker_456" in joined


class TestTranscript:
    def test_splits_on_speaker_turns(self):
        text = (
            "Ansem: " + "solana stuff " * 30 + "\n"
            "Banks: " + "creator stuff " * 30 + "\n"
            "Ansem: short reply\n"
        )
        chunks = chunk_transcript(doc(SourceType.PODCAST, text), MAX, OVERLAP)
        assert chunks, "transcript produced no chunks"
        # A speaker's utterance is never severed from its label.
        for c in chunks:
            assert not c.text.startswith("stuff")

    def test_fallback_without_speaker_labels(self):
        text = "para one\n\npara two\n\npara three"
        chunks = chunk_transcript(doc(SourceType.PODCAST, text), MAX, OVERLAP)
        assert len(chunks) == 1
        assert "para two" in chunks[0].text


class TestTweets:
    def test_one_chunk_per_tweet(self):
        text = "first tweet about $ANSEM\n\nsecond tweet about perps"
        chunks = chunk_tweets(doc(SourceType.TWEET, text), MAX, OVERLAP)
        assert len(chunks) == 2
        assert chunks[0].text == "first tweet about $ANSEM"

    def test_blank_segments_dropped(self):
        chunks = chunk_tweets(doc(SourceType.TWEET, "a\n\n\n\n\n\nb"), MAX, OVERLAP)
        assert len(chunks) == 2


class TestMetadataTagging:
    def test_chunks_carry_full_metadata(self):
        from app.ingest import IngestionPipeline

        pipeline = IngestionPipeline()
        document = IngestDocument(
            source_type=SourceType.PODCAST,
            source_id="ep-12",
            title="Market Bubble Ep 12",
            author="Ansem",
            url="https://example.com/ep12",
            published_at="2026-07-01",
            text="Ansem: " + "alpha " * 500 + "\nBanks: " + "beta " * 500,
        )
        chunks = pipeline._chunk(document)
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks):
            md = chunk.metadata
            assert md["source_type"] == "podcast"
            assert md["source_id"] == "ep-12"
            assert md["chunk_index"] == i
            assert md["title"] == "Market Bubble Ep 12"
            assert md["text"] == chunk.text  # text round-trips for retrieval

    def test_absent_optional_fields_are_omitted(self):
        from app.ingest import IngestionPipeline

        pipeline = IngestionPipeline()
        document = IngestDocument(
            source_type=SourceType.DOCS, source_id="d1", text="# T\nbody"
        )
        md = pipeline._chunk(document)[0].metadata
        # Pinecone rejects null metadata values — they must be absent.
        for key in ("title", "author", "url", "published_at"):
            assert key not in md

    def test_settings_defaults_sane(self):
        s = get_settings()
        assert s.chunk_overlap_chars < s.chunk_max_chars


class TestQueryEmbeddingCache:
    def test_repeat_query_hits_cache(self):
        import asyncio

        from app import embeddings

        embeddings._QUERY_CACHE.clear()
        calls = {"n": 0}

        class FakeVoyage:
            async def embed(self, texts, **k):
                calls["n"] += 1

                class R:
                    embeddings = [[0.1] * 8 for _ in texts]
                return R()

        async def run():
            fv = FakeVoyage()
            a = await embeddings.embed_query(fv, "same q", model="m", dimension=8)
            b = await embeddings.embed_query(fv, "same q", model="m", dimension=8)
            return a, b

        a, b = asyncio.run(run())
        assert a == b
        assert calls["n"] == 1, "second identical query must be served from cache"
        embeddings._QUERY_CACHE.clear()
