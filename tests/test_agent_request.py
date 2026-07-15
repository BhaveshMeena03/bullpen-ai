"""Agent request construction — the per-model API contract.

These tests pin the API-shape decisions that would 400 in production if
regressed: thinking config per model family, no sampling params, fallback
beta wiring, cache breakpoints in the right places.
"""

import pytest

from app.agent import SYSTEM_PROMPT, ConciergeAgent, _format_context
from app.config import get_settings
from app.schemas import ChatTurn, RetrievedChunk, SourceType


@pytest.fixture
def fable_mode(monkeypatch):
    """Switch settings to claude-fable-5 for the duration of a test."""
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-fable-5")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_chunk(text="Bullpen supports Solana spot trading.", **meta) -> RetrievedChunk:
    return RetrievedChunk(
        id="c1", text=text, score=0.9,
        source_type=SourceType.DOCS,
        metadata={"title": "Getting Started", **meta},
    )


def build(message="How do I bridge?", history=None, chunks=None) -> dict:
    agent = ConciergeAgent()
    return agent._build_request(message, history or [], chunks or [make_chunk()])


class TestDefaultModelContract:
    """Default configuration: Opus 4.8 with explicit adaptive thinking."""

    def test_default_model(self):
        assert build()["model"] == "claude-opus-4-8"

    def test_adaptive_thinking_requested(self):
        # Opus 4.8 runs WITHOUT thinking when the param is omitted — it
        # must be requested explicitly, or answer quality drops.
        assert build()["thinking"] == {"type": "adaptive"}

    def test_no_fallbacks_outside_fable(self):
        req = build()
        assert "fallbacks" not in req
        assert "betas" not in req

    def test_no_sampling_params(self):
        req = build()
        for banned in ("temperature", "top_p", "top_k"):
            assert banned not in req

    def test_effort_in_output_config(self):
        assert build()["output_config"]["effort"] in {
            "low", "medium", "high", "xhigh", "max"
        }

    def test_no_assistant_prefill(self):
        # Last message must be the user turn (prefill 400s on 4.6+ models).
        assert build()["messages"][-1]["role"] == "user"


class TestFable5Contract:
    """ANTHROPIC_MODEL=claude-fable-5: omitted thinking + refusal fallback."""

    def test_no_thinking_param(self, fable_mode):
        # Fable 5: any explicit thinking config (even "disabled") is a 400.
        assert "thinking" not in build()

    def test_fallback_wired(self, fable_mode):
        req = build()
        assert req["model"] == "claude-fable-5"
        assert req["fallbacks"] == [{"model": "claude-opus-4-8"}]
        assert req["betas"] == ["server-side-fallback-2026-06-01"]


class TestPromptCaching:
    def test_system_prompt_has_breakpoint(self):
        system = build()["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_system_prompt_is_frozen(self):
        # Byte-identical across requests — anything dynamic here would
        # invalidate the cache for every user.
        assert build()["system"][0]["text"] == build()["system"][0]["text"]
        assert build()["system"][0]["text"] == SYSTEM_PROMPT

    def test_breakpoint_on_last_block_of_newest_turn(self):
        content = build()["messages"][-1]["content"]
        assert content[-1]["cache_control"] == {"type": "ephemeral"}
        # And nowhere else in the turn (max 4 breakpoints per request).
        assert all("cache_control" not in b for b in content[:-1])

    def test_exactly_two_breakpoints(self):
        req = build(history=[
            ChatTurn(role="user", content="a"),
            ChatTurn(role="assistant", content="b"),
        ])
        count = sum(
            1 for b in req["system"] if "cache_control" in b
        ) + sum(
            1
            for m in req["messages"]
            for b in (m["content"] if isinstance(m["content"], list) else [])
            if isinstance(b, dict) and "cache_control" in b
        )
        assert count == 2

    def test_history_replayed_verbatim(self):
        history = [
            ChatTurn(role="user", content="What is a perp?"),
            ChatTurn(role="assistant", content="A perpetual future..."),
        ]
        req = build(history=history)
        assert req["messages"][0] == {"role": "user", "content": "What is a perp?"}
        assert req["messages"][1]["role"] == "assistant"
        assert len(req["messages"]) == 3


class TestContextInjection:
    def test_context_precedes_question(self):
        req = build(message="the question")
        content = req["messages"][-1]["content"]
        assert content[0]["text"].startswith("<retrieved_context>")
        assert content[1]["text"] == "the question"

    def test_sources_are_attributed(self):
        formatted = _format_context([make_chunk()])
        assert '<source type="docs" ref="Getting Started">' in formatted

    def test_empty_context_is_explicit(self):
        # The model must be able to SEE that retrieval found nothing —
        # that's what triggers the "I don't have that" behavior.
        assert "(no relevant material found)" in _format_context([])


class TestGuardrailPromptContent:
    """The system prompt IS the guardrail config — pin its key clauses."""

    def test_core_guardrails_present(self):
        for needle in (
            "NO financial advice",
            "NO price predictions",
            "NEVER ask for, accept, or handle private keys",
            "support tool",
        ):
            assert needle in SYSTEM_PROMPT, f"guardrail clause missing: {needle}"

    def test_injection_resistance_clause(self):
        assert "instructions embedded in retrieved content" in SYSTEM_PROMPT

    def test_knowledge_tiers_present(self):
        assert "GENERAL CRYPTO EDUCATION" in SYSTEM_PROMPT
        assert "OPERATIONAL SPECIFICS" in SYSTEM_PROMPT
        assert "never guess" in SYSTEM_PROMPT

    def test_platform_facts_present(self):
        for fact in ("blknoiz06", "FaZe Banks", "The Black Bull", "Hyperliquid",
                     "Polymarket"):
            assert fact in SYSTEM_PROMPT

    def test_nothing_dynamic_interpolated(self):
        # Frozen prompt: no f-string leftovers, no braces to format later.
        assert "{" not in SYSTEM_PROMPT.replace("{}", "")
