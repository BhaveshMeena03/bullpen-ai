"""Shared fixtures. Dummy credentials let every module import and
construct clients without touching real services (all SDK clients here
are lazy — no network happens until a call is made)."""

import os
import sys
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("VOYAGE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Rate-limit buckets are module-level state; clear them so tests are
    isolated (every TestClient shares one 'testclient' IP)."""
    from app.security import global_rate_limit, public_rate_limit

    public_rate_limit._buckets.clear()
    global_rate_limit._buckets.clear()
    yield
