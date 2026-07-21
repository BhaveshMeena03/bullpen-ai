"""The concierge knowledge-base data files must always be well-formed.

seed.json (ecosystem) and bullpen_docs.json (official Bullpen docs, produced
by scripts/fetch_bullpen_docs.py) are ingested verbatim, so a malformed file
or a bad fetch would silently degrade the agent. These checks fail loudly
instead.
"""

import json
from pathlib import Path

import pytest

from app.schemas import IngestDocument

DATA = Path(__file__).resolve().parent.parent / "data"


def _load(name: str) -> list[dict]:
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", ["seed.json", "bullpen_docs.json"])
def test_kb_file_is_valid_ingest_documents(name):
    raw = _load(name)
    assert raw, f"{name} is empty"
    docs = [IngestDocument(**d) for d in raw]          # validates the shape
    ids = [d.source_id for d in docs]
    assert len(ids) == len(set(ids)), f"{name} has duplicate source_ids"
    for d in docs:
        assert d.text.strip(), f"{d.source_id} has empty text"


def test_official_docs_are_cleaned_of_gitbook_syntax():
    """fetch_bullpen_docs.py must strip GitBook authoring tags; leftover
    `{% ... %}` would render as noise in a cited answer."""
    for d in _load("bullpen_docs.json"):
        assert "{%" not in d["text"], f"{d['source_id']} still has GitBook tags"
        assert "<figure>" not in d["text"]


def test_seed_is_ecosystem_only_not_placeholder_operational():
    """The official docs are the operational source of truth now; seed.json
    should no longer carry the old hand-written operational placeholders."""
    ids = {d["source_id"] for d in _load("seed.json")}
    removed = {"funding-wallet", "spot-trading", "perps-trading",
               "connect-wallet", "fees-and-official-sources", "common-errors"}
    assert not (ids & removed), f"placeholder operational docs still present: {ids & removed}"
