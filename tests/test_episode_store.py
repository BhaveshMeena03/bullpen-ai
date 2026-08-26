"""Concurrent writes to episodes.json must not lose each other's work.

The bug this replaces was not hypothetical. Every fetcher read the file at
startup, worked for minutes or hours, and wrote its own copy back at exit,
silently reverting anything that finished in between — an X clip fetched
during a long transcription disappeared exactly that way.

It stops being an edge case under the intended workflow: an episode lands
on X and is transcribed by hand, then lands on YouTube a day later and is
picked up by a scheduled job. Overlap is the normal case there.

These tests run real processes against a real file, because the failure is
about what two writers do to each other and a mocked lock proves nothing.
"""

import json
import multiprocessing as mp
import time

import pytest

from app.episode_store import load, merge


def episode(episode_id: str, title: str = "t", segments: int = 2) -> dict:
    return {
        "episode_id": episode_id,
        "title": title,
        "url": f"https://www.youtube.com/watch?v={episode_id}",
        "platform": "youtube",
        "published_at": "2026-08-01",
        "segments": [{"t": float(i), "text": f"line {i}"} for i in range(segments)],
    }


def test_merge_creates_the_file(tmp_path):
    path = tmp_path / "episodes.json"
    added, updated = merge([episode("a")], path)
    assert (added, updated) == (1, 0)
    assert [e["episode_id"] for e in load(path)] == ["a"]


def test_merge_adds_without_dropping_what_is_there(tmp_path):
    path = tmp_path / "episodes.json"
    merge([episode("a")], path)
    merge([episode("b")], path)
    assert sorted(e["episode_id"] for e in load(path)) == ["a", "b"]


def test_same_id_updates_rather_than_duplicating(tmp_path):
    path = tmp_path / "episodes.json"
    merge([episode("a", title="first")], path)
    added, updated = merge([episode("a", title="second")], path)
    assert (added, updated) == (0, 1)
    stored = load(path)
    assert len(stored) == 1
    assert stored[0]["title"] == "second"


def test_replace_all_rewrites_everything(tmp_path):
    """The cleanup scripts rewrite every episode rather than contributing
    new ones, and must still hold the lock while doing it."""
    path = tmp_path / "episodes.json"
    merge([episode("a"), episode("b")], path)
    merge([episode("a", title="only one left")], path, replace_all=True)
    assert [e["episode_id"] for e in load(path)] == ["a"]


def test_refuses_to_overwrite_an_unreadable_file(tmp_path):
    """A corrupt file that gets replaced by one episode is how a catalogue
    disappears. Better to fail loudly than to start from nothing."""
    path = tmp_path / "episodes.json"
    path.write_text("{ this is not json")
    with pytest.raises(RuntimeError):
        merge([episode("a")], path)


def test_sorted_newest_first(tmp_path):
    path = tmp_path / "episodes.json"
    old, new = episode("old"), episode("new")
    old["published_at"], new["published_at"] = "2026-01-01", "2026-09-09"
    merge([old, new], path)
    assert [e["episode_id"] for e in load(path)] == ["new", "old"]


# --- the actual point ------------------------------------------------------

def _slow_writer(path_str: str, episode_id: str, hold: float) -> None:
    """Read early, work a while, write late — the old, broken shape."""
    from pathlib import Path

    from app.episode_store import merge as m
    time.sleep(hold)
    m([episode(episode_id)], Path(path_str))


def test_two_overlapping_writers_both_survive(tmp_path):
    """The regression test for the bug that ate a fetch.

    Both processes start while the file holds only "existing". Under the
    old read-at-start/write-at-end pattern the later writer would clobber
    the earlier one and the file would end with two episodes instead of
    three. Merging re-reads inside the lock, so both land.
    """
    path = tmp_path / "episodes.json"
    merge([episode("existing")], path)

    ctx = mp.get_context("spawn")   # fork is unsafe under pytest on macOS
    procs = [ctx.Process(target=_slow_writer, args=(str(path), name, delay))
             for name, delay in (("fast", 0.0), ("slow", 0.4))]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    assert sorted(e["episode_id"] for e in load(path)) == [
        "existing", "fast", "slow"]


def test_many_concurrent_writers_all_land(tmp_path):
    """Eight at once, which no real workflow needs — but if any write can be
    lost, contention is where it shows."""
    path = tmp_path / "episodes.json"
    merge([episode("seed")], path)

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_slow_writer,
                         args=(str(path), f"e{i}", 0.05 * i))
             for i in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    stored = {e["episode_id"] for e in load(path)}
    assert stored == {"seed"} | {f"e{i}" for i in range(8)}


def test_a_reader_never_sees_a_half_written_file(tmp_path):
    """Writes land by atomic rename, so a reader gets the old file or the
    new one — never a truncated one. Read while writing and check the JSON
    always parses."""
    path = tmp_path / "episodes.json"
    merge([episode(f"e{i}", segments=200) for i in range(20)], path)

    ctx = mp.get_context("spawn")
    writer = ctx.Process(target=_slow_writer, args=(str(path), "late", 0.1))
    writer.start()
    for _ in range(60):
        text = path.read_text()
        json.loads(text)            # raises if a partial file is ever visible
        time.sleep(0.01)
    writer.join(timeout=60)
    assert writer.exitcode == 0


def _names_bound_to_episodes(tree) -> set:
    """Module-level names assigned a path that ends in episodes.json."""
    import ast
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (isinstance(target, ast.Name)
                    and "episodes.json" in ast.unparse(node.value)):
                found.add(target.id)
    return found


def test_no_script_writes_the_episode_file_directly():
    """The lock only works if everyone takes it.

    One script that still calls write_text on episodes.json reintroduces the
    whole bug, and it would do so silently — the file stays valid JSON, just
    missing whatever the other process had added. Cheaper to fail here than
    to notice an episode is gone weeks later.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    scripts = sorted((root / "scripts").glob("*.py"))
    trees = {s.name: ast.parse(s.read_text()) for s in scripts}

    # Which name means "the episode file" in each module. The name alone is
    # not enough to go on: extract_assets.py has an OUT and it is assets.json.
    owned = {name: _names_bound_to_episodes(tree) for name, tree in trees.items()}

    # ...and sync_latest.py, the cron, does not define OUT at all — it
    # imports it from fetch_episodes. Following imports is what makes this
    # test able to fail; without it the most important writer is skipped.
    for name, tree in trees.items():
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    ("scripts.", "scripts")):
                source = (node.module or "").split(".")[-1] + ".py"
                for alias in node.names:
                    if alias.name in owned.get(source, set()):
                        owned[name].add(alias.asname or alias.name)

    # Names that stand in for the episode file inside a function: argparse
    # destinations and the parameters the cleanup scripts take.
    indirect = {"out_path", "args.file", "path"}
    offenders = []

    for script in scripts:
        episode_names = owned[script.name]
        if not episode_names:
            continue
        for node in ast.walk(trees[script.name]):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write_text"):
                continue
            written = ast.unparse(node.func.value)
            if written in episode_names or written in indirect:
                offenders.append(
                    f"{script.name}:{node.lineno}  {written}.write_text(...)")

    assert not offenders, (
        "these write the episode file without the lock:\n  "
        + "\n  ".join(offenders))
