"""Safe concurrent access to data/episodes.json.

Every fetcher used to read the file at startup, work for minutes or hours,
and write its own copy back at exit. Whatever else happened in between was
silently reverted. That is not theoretical: an X clip fetched successfully
in the middle of a long transcription simply vanished, and it was noticed
only because the episode count looked wrong afterwards.

It becomes routine rather than rare under the intended workflow — an
episode lands on X and is transcribed by hand, then lands on YouTube a day
later and is picked up by a scheduled job. Those two overlapping is the
normal case, not the accident.

So writes go through `merge`, which takes an exclusive lock, re-reads the
file INSIDE the lock, merges, and replaces atomically. The re-read is the
part that matters: a snapshot taken before a two-hour transcription is
stale by definition, and merging against it would reintroduce exactly the
bug. Whoever finishes last sees everything that finished before them.

Reads are not locked. A reader can catch a slightly old file, never a
half-written one, because os.replace is atomic.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / "data" / "episodes.json"

# Long enough that a slow transcription's final write is not refused while
# another process holds the lock, short enough that a crashed holder does
# not wedge a scheduled job forever. The lock is only held across the merge
# itself — reading, combining and writing a few megabytes — never across
# downloading or transcribing.
LOCK_TIMEOUT_SECONDS = 120


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on a sidecar file for `path`.

    A separate .lock file rather than the data file itself, because the data
    file is replaced by rename: locking an inode that is about to be
    unlinked would let a second writer take a lock on the new file while the
    first still believes it holds one.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    with open(lock_path, "w") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"another process has held {lock_path.name} for "
                        f"{LOCK_TIMEOUT_SECONDS}s; is a fetch stuck?"
                    ) from None
                time.sleep(0.5)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load(path: Path = EPISODES) -> list[dict]:
    """Every episode on file, or an empty list if there is no file yet."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # Never silently start from nothing: an unreadable file that gets
        # overwritten with one episode is how a catalogue disappears.
        raise RuntimeError(
            f"{path} is not valid JSON — refusing to overwrite it"
        ) from None


def _write_atomic(episodes: list[dict], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(episodes, ensure_ascii=False))
    os.replace(tmp, path)          # atomic: a reader sees old or new, never half


def merge(new: list[dict], path: Path = EPISODES,
          replace_all: bool = False) -> tuple[int, int]:
    """Add or update episodes without losing concurrent work.

    Returns (added, updated). With `replace_all` the given list becomes the
    entire file — used by the cleanup scripts, which rewrite every episode
    rather than contributing new ones, and which still take the lock so they
    cannot interleave with a fetch.
    """
    with _exclusive(path):
        if replace_all:
            _write_atomic(new, path)
            return 0, len(new)

        current = load(path)
        by_id = {e["episode_id"]: e for e in current}
        added = updated = 0
        for episode in new:
            if episode["episode_id"] in by_id:
                updated += 1
            else:
                added += 1
            by_id[episode["episode_id"]] = episode

        merged = sorted(by_id.values(),
                        key=lambda e: (e.get("published_at") or ""),
                        reverse=True)
        _write_atomic(merged, path)
        if added or updated:
            logger.info("episodes.json: +%d new, %d updated, %d total",
                        added, updated, len(merged))
        return added, updated
