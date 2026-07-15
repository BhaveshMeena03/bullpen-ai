"""Demo mock server: keyword routing must send each question type to the
right canned answer (this is what runs in pitch demos — it must not
misfire on the questions people will actually ask)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from mock_server import CANNED, pick_answer  # noqa: E402

CASES = [
    ("who is ansem", "ansem_person"),
    ("Who is Ansem?", "ansem_person"),
    ("who is banks", "banks"),
    ("who is faze banks?", "banks"),
    ("what is bullpen", "bullpen"),
    ("What is Bullpen?", "bullpen"),
    ("what is $ansem", "ansem_token"),
    ("what is the ansem coin", "ansem_token"),
    ("How do I claim the $ANSEM airdrop?", "airdrop"),
    ("am i eligible for the airdrop", "airdrop"),
    ("What is a perp?", "perp"),
    ("how does leverage work", "perp"),
    ("what happens at liquidation", "perp"),
    ("what is a wallet", "wallet"),
    ("what is a seed phrase", "wallet"),
    ("should i use phantom", "wallet"),
    ("what is a blockchain", "default"),
]


def test_routing_table():
    failures = []
    for question, expected in CASES:
        got = pick_answer(question)
        if got != CANNED[expected]:
            actual = next(k for k, v in CANNED.items() if v == got)
            failures.append(f"{question!r} -> {actual} (expected {expected})")
    assert not failures, "\n".join(failures)


def test_every_canned_answer_reachable():
    reached = {pick_answer(q) for q, _ in CASES}
    for key, text in CANNED.items():
        assert text in reached, f"canned answer {key!r} is unreachable"


def test_no_financial_advice_in_canned_answers():
    for key, text in CANNED.items():
        lowered = text.lower()
        for banned in ("you should buy", "will go up", "price target", "moon"):
            assert banned not in lowered, f"{key} contains advice-like text"
