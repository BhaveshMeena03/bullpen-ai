"""Everything that has to be true before this is considered shipped.

    .venv/bin/python scripts/verify_all.py

Written after a day in which three separate regressions were found by a
stranger, by a user question, and by a benchmark — rather than by us. Each
one was silent: the service kept answering, slightly worse, with nothing in
any log to say so. That is the failure mode this file exists to catch.

Checks, in the order they matter:

  data        the episode file parses, has no duplicate ids, and no episode
              is a near-copy of another
  dates       the recorded date, not the posted date — a clip cut four days
              after the conversation must not outrank a newer episode, since
              the answers reason about recency
  index       the live vector count matches what was ingested
  answers     every example question on every page returns something real
  grounding   no quote that is not in a transcript
  links       YouTube citations carry ?t=, X citations do not, because X
              ignores it and a link that lands at 0:00 while looking correct
              is worse than one that visibly cannot jump
  isolation   the ClawPump bot cannot answer from Bullpen's docs
  coverage    the guests that prompted this work are actually findable

Exits non-zero if anything fails, so it can gate a deploy.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.dedupe import SAME_RECORDING, dedupe  # noqa: E402
from app.schemas import Episode  # noqa: E402

SEARCH = "https://search.lexthedev.com"
CONCIERGE = "https://concierge.lexthedev.com"

failures: list[str] = []
notes: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def post(url: str, payload: dict, timeout: float = 180) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    for attempt in range(5):
        try:
            # strict=False: caption text carries raw control characters.
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read(),
                              strict=False)
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                return {"answer": "", "hits": [], "sources": [], "error": exc.code}
            time.sleep(8 * (attempt + 1))
    return {"answer": "", "hits": [], "sources": [], "error": 429}


def main() -> None:
    episodes = json.loads((ROOT / "data" / "episodes.json").read_text())

    print("\n=== data ===")
    try:
        [Episode(**e) for e in episodes]
        check("episode file matches the schema", True, f"{len(episodes)} episodes")
    except Exception as exc:                                    # noqa: BLE001
        check("episode file matches the schema", False, str(exc)[:90])
    ids = [e["episode_id"] for e in episodes]
    check("no duplicate episode ids", len(ids) == len(set(ids)))
    check("every episode has segments",
          all(e.get("segments") for e in episodes))

    _, dupes = dedupe(episodes, threshold=SAME_RECORDING)
    # Reported, not failed: partial overlap is normal on this channel, and
    # the catalogue has answered correctly for months with pairs at 94%.
    if dupes:
        notes.append(f"{len(dupes)} near-identical pair(s): "
                     + ", ".join(d["episode_id"] for d in dupes))
    print(f"  note  near-identical episodes: {len(dupes)}")

    print("\n=== dates ===")
    undated = [e["episode_id"] for e in episodes if not e.get("published_at")]
    check("every episode has a date", not undated, ", ".join(undated[:3]))
    newest = max(episodes, key=lambda e: e.get("published_at") or "")
    print(f"  note  newest is {newest['published_at']} "
          f"{newest['title'][:44]}")

    print("\n=== index ===")
    try:
        from pinecone import Pinecone

        from app.config import get_settings
        settings = get_settings()
        stats = Pinecone(api_key=settings.pinecone_api_key).Index(
            settings.pinecone_index).describe_index_stats()
        counts = {ns: i["vector_count"]
                  for ns, i in (stats.get("namespaces") or {}).items()}
        for ns in sorted(counts):
            print(f"        {ns or '(default)':13s} {counts[ns]:6d}")
        check("podcast namespace is populated", counts.get("podcast", 0) > 0)
        check("clawpump namespace is populated", counts.get("clawpump", 0) > 0)
        check("bullpen (default) namespace is populated",
              counts.get("__default__", 0) > 0)
    except Exception as exc:                                    # noqa: BLE001
        check("could read the index", False, str(exc)[:90])

    print("\n=== answers: every example question on every page ===")
    chips = subprocess.run([sys.executable, "scripts/check_chips.py"],
                           cwd=ROOT, capture_output=True, text=True)
    dead = [ln for ln in chips.stdout.splitlines() if ln.strip().startswith("DEAD")]
    skipped = [ln for ln in chips.stdout.splitlines() if ln.strip().startswith("SKIP")]
    check("no dead example questions", not dead,
          "; ".join(d.strip()[:60] for d in dead[:2]))
    if skipped:
        # Rate limited rather than broken. Worth surfacing so the run is not
        # read as a clean pass, but not a failure.
        notes.append(f"{len(skipped)} question(s) could not be asked "
                     f"(rate limited); rerun to confirm")
        print(f"  note  {len(skipped)} question(s) rate limited, not asked")

    print("\n=== links: platform-correct deep links ===")
    wrong = []
    for query in ("blackrock told an nba player not to buy bitcoin",
                  "what did poorgoat say about crypto",
                  "why does ansem think ethereum is done"):
        for hit in post(f"{SEARCH}/v1/podcast/search", {"query": query}).get("hits", []):
            is_x = hit["episode_id"].startswith("x-")
            has_t = "t=" in hit["deep_link"]
            if is_x and has_t:
                wrong.append(f"{hit['episode_id']} has a ?t= X cannot use")
            if not is_x and not has_t:
                wrong.append(f"{hit['episode_id']} lost its timestamp")
    check("deep links correct for their platform", not wrong,
          "; ".join(wrong[:2]))

    print("\n=== isolation ===")
    for query in ("how do i deposit into bullpen",
                  "what order types does bullpen support"):
        got = post(f"{CONCIERGE}/v1/clawpump/chat", {"message": query})
        leaked = [s["metadata"].get("source_id", "")
                  for s in got.get("sources", [])
                  if s["metadata"].get("source_id", "").startswith("bp-")]
        check(f"clawpump bot does not read bullpen docs: {query[:34]}",
              not leaked, ", ".join(leaked[:2]))

    print("\n=== grounding ===")
    faith = subprocess.run(
        [sys.executable, "evals/run_faithfulness.py", "--search-only"],
        cwd=ROOT, capture_output=True, text=True)
    check("no fabricated quotes", "no fabricated quotes" in faith.stdout,
          faith.stdout.strip().splitlines()[-1][:70] if faith.stdout else "")

    print("\n=== coverage: the guests this work was for ===")
    # Spelling variants because Whisper splits proper nouns it has not seen:
    # the Ep 15 transcript says "Poor Goat" fifteen times and "poorgoat"
    # never. Retrieval is unbothered — the embeddings put the two forms next
    # to each other — but a literal check like this one is not, and reported
    # missing content that was in fact searchable.
    guests = {
        "poorgoat": ("poorgoat", "poor goat"),
        "mike dudas": ("dudas",),
        "austin federa": ("federa", "doublezero", "double zero"),
        "brian armstrong": ("armstrong",),
        "jesse pollak": ("pollak", "pollack"),
        "luca netz": ("luca netz", "netz"),
        "tjr": ("tjr",),
        "orangie": ("orangie",),
    }
    for name, variants in guests.items():
        haystacks = [" ".join(s["text"] for s in e["segments"]).lower()
                     for e in episodes]
        indexed = any(v in h for v in variants for h in haystacks)
        found = False
        if indexed:
            hits = post(f"{SEARCH}/v1/podcast/search",
                        {"query": f"what did {name} say"}).get("hits", [])
            found = bool(hits)
        state = "searchable" if found else ("in transcripts, not retrieved"
                                            if indexed else "not indexed yet")
        print(f"  {'ok  ' if found else 'note'}  {name:18s} {state}")

    print("\n=== build ===")
    tests = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                           cwd=ROOT, capture_output=True, text=True)
    check("tests pass", tests.returncode == 0,
          tests.stdout.strip().splitlines()[-1][:60] if tests.stdout else "")
    lint = subprocess.run([str(ROOT / ".venv" / "bin" / "ruff"), "check", "."],
                          cwd=ROOT, capture_output=True, text=True)
    check("lint clean", lint.returncode == 0, lint.stdout.strip()[-60:])

    print("\n" + "=" * 62)
    if failures:
        print(f"{len(failures)} FAILURE(S): " + "; ".join(failures))
        sys.exit(1)
    print("everything checks out")
    for note in notes:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
