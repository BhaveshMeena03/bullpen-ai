"""The ClawPump support surface, and the isolation it depends on.

The interesting risk here is not that an answer is wrong — it is that an
answer is right *about the wrong product*. This service runs two support
bots over one Pinecone index, for two Solana platforms whose docs both talk
about fees, wallets, launching and execution. What keeps them apart is a
namespace, and a namespace is one keyword argument that a refactor can drop
without any test noticing. These are the tests that would notice.
"""

import json
from pathlib import Path

import pytest

from app.clawpump import NAMESPACE, SYSTEM_PROMPT, ClawPumpAgent
from app.schemas import ChatRequest, IngestDocument

DATA = Path(__file__).resolve().parent.parent / "data"


def _docs() -> list[dict]:
    path = DATA / "clawpump_docs.json"
    if not path.exists():
        pytest.skip("clawpump_docs.json not present")
    return json.loads(path.read_text())


# --- the knowledge base ----------------------------------------------------

def test_kb_is_valid_ingest_documents():
    raw = _docs()
    assert raw, "clawpump_docs.json is empty"
    docs = [IngestDocument(**d) for d in raw]
    assert all(d.text.strip() for d in docs)
    assert len({d.source_id for d in docs}) == len(docs), "duplicate source_id"


def test_kb_kept_heading_structure():
    """Chunking splits on Markdown headings.

    These pages are scraped from rendered HTML, so if the converter ever
    regresses to plain text the file still looks fine — same pages, similar
    length — while every page collapses into one undifferentiated chunk and
    retrieval quietly gets worse. Headings are the thing to assert on.
    """
    docs = [IngestDocument(**d) for d in _docs()]
    with_headings = [d for d in docs if "\n#" in d.text or d.text.startswith("#")]
    assert len(with_headings) >= len(docs) * 0.8


def test_kb_covers_the_pages_people_actually_ask_about():
    """Guards the failure this fetcher was written to avoid.

    The Bullpen fetcher began with a hand-written page list, went stale, and
    left that bot unable to answer most of what it was asked. Discovery here
    unions the sitemap, llms.txt and a link crawl precisely because each of
    those alone was missing something: /docs-x is absent from the sitemap,
    and /terms and /privacy are absent from both indexes.
    """
    ids = {d["source_id"] for d in _docs()}
    for required in ("cp-docs", "cp-docs-x", "cp-guide", "cp-developers",
                     "cp-terms", "cp-privacy", "cp-ansemhack"):
        assert required in ids, f"{required} missing from the knowledge base"


def test_kb_excludes_live_listings():
    """Token pages and leaderboards are market data, not documentation.

    Embedding them puts numbers into the index that are wrong within the
    hour, and the bot would state them in a support voice.
    """
    ids = {d["source_id"] for d in _docs()}
    assert not any(i.startswith("cp-tokens-") for i in ids)
    for banned in ("cp-leaderboard", "cp-analytics", "cp-marketplace"):
        assert banned not in ids


# --- isolation -------------------------------------------------------------

def test_namespace_is_not_the_default():
    """The default namespace holds the Bullpen concierge. Sharing it would
    let either bot answer from the other's documentation."""
    assert NAMESPACE and NAMESPACE not in ("", "__default__")


@pytest.mark.anyio
async def test_search_is_scoped_to_the_clawpump_namespace():
    """Both ClawPump routes must pass the namespace to the retriever."""
    from app import main

    seen: dict = {}

    class FakeRetriever:
        async def search(self, query, filters=None, top_k=None, namespace=None):
            seen["namespace"] = namespace
            seen["filters"] = filters
            return []

    class FakeAgent:
        async def answer(self, message, history, chunks, brief=False):
            from app.schemas import ChatResponse
            return ChatResponse(answer="ok", sources=[], model="test")

    await main.clawpump_chat(
        ChatRequest(message="what are the fees"),
        retriever=FakeRetriever(), agent=FakeAgent(),
    )
    assert seen["namespace"] == NAMESPACE
    # Caller-supplied filters must not reach this route: on a support bot the
    # searched corpus is not something the client gets to influence.
    assert seen["filters"] is None


@pytest.mark.anyio
async def test_caller_cannot_redirect_the_search_with_filters():
    from app import main

    seen: dict = {}

    class FakeRetriever:
        async def search(self, query, filters=None, top_k=None, namespace=None):
            seen.update(namespace=namespace, filters=filters)
            return []

    class FakeAgent:
        async def answer(self, message, history, chunks, brief=False):
            from app.schemas import ChatResponse
            return ChatResponse(answer="ok", sources=[], model="test")

    await main.clawpump_chat(
        ChatRequest(message="hi", filters={"source_type": "podcast"}),
        retriever=FakeRetriever(), agent=FakeAgent(),
    )
    assert seen["namespace"] == NAMESPACE
    assert seen["filters"] is None


# --- the prompt ------------------------------------------------------------

def test_agent_uses_its_own_prompt_not_the_bullpen_one():
    from app.agent import SYSTEM_PROMPT as BULLPEN_PROMPT

    assert ClawPumpAgent.system_prompt == SYSTEM_PROMPT
    assert ClawPumpAgent.system_prompt != BULLPEN_PROMPT
    assert "Bullpen" not in SYSTEM_PROMPT


def test_prompt_states_it_is_unofficial():
    """It answers about someone else's product using their branding. Users
    have to be able to tell it is not the vendor's own support."""
    assert "NOT operated by the ClawPump team" in SYSTEM_PROMPT


def test_prompt_carries_the_published_scam_domain_warning():
    """ClawPump publishes that clawpump.tech is its only official site.
    A support bot for an audience being phished during a hackathon should
    repeat that, and must never bless a lookalike domain."""
    assert "clawpump.net" in SYSTEM_PROMPT
    assert "clawpumpsol.com" in SYSTEM_PROMPT
    assert "clawpump.tech" in SYSTEM_PROMPT


def test_prompt_keeps_the_core_guardrails():
    for rule in ("NO financial advice", "NO price predictions",
                 "NEVER ask for, accept or handle private keys"):
        assert rule in SYSTEM_PROMPT
