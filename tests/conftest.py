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
