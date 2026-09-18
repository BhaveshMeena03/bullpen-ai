"""The two host voices are named by what they said, not by rank.

Names used to follow how many episodes each voice recurred in. Both hosts
are on every show, so the counts converged, tied at 37, and adding ep 20
flipped the order: all 40,263 labels in the archive swapped hands, and
eps 1, 13 and 20 went live with Banks's lines under Ansem's name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_speaker_map import ANCHOR_SPEAKER, name_by_anchor  # noqa: E402


def _unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=float)
    return v / np.linalg.norm(v)


BANKS = _unit(1.0, 0.1, 0.0)
ANSEM = _unit(0.1, 1.0, 0.0)


def _names(voices, anchor):
    return {name: centroid.tobytes() for name, centroid, _ in
            name_by_anchor(voices, anchor)}


def test_the_anchor_line_decides_which_voice_is_banks():
    named = _names([(BANKS, 37), (ANSEM, 37)], BANKS)
    assert named[ANCHOR_SPEAKER] == BANKS.tobytes()
    assert named["Ansem"] == ANSEM.tobytes()


def test_the_order_the_voices_arrive_in_changes_nothing():
    """The failure itself: a tie reordered the voices, and the names moved."""
    one = _names([(BANKS, 37), (ANSEM, 37)], BANKS)
    two = _names([(ANSEM, 37), (BANKS, 37)], BANKS)
    assert one == two


def test_refuses_when_the_anchor_matches_neither_voice_clearly():
    between = _unit(1.0, 1.0, 0.0)
    with pytest.raises(SystemExit):
        name_by_anchor([(BANKS, 37), (ANSEM, 37)], between)


def test_refuses_without_exactly_two_voices():
    with pytest.raises(SystemExit):
        name_by_anchor([(BANKS, 37)], BANKS)
