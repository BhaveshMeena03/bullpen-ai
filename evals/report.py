"""Turn eval results into a documentation-gap report.

    python evals/report.py [--in PATH] [--out PATH]

The interesting output is not the score. It is the list of questions a
real user would ask that the documentation cannot answer — ranked by
section, so it reads as a work list rather than a grade.

Reading the three groups:
  answerable  a gap here is a retrieval failure, ours to fix
  edge        a gap here is a documentation hole, theirs to fill
  guardrail   anything not refused is a safety regression, and it
              invalidates the rest of the report until it's fixed
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.classify import classify  # noqa: E402

HERE = Path(__file__).resolve().parent
GAP = {"no_context", "low_confidence"}


def load(path: Path) -> list[dict]:
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    for r in rows:
        r["outcome"] = classify(r.get("answer", ""), r.get("n_sources", 0),
                                r.get("refused", False))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, default=HERE / "results.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "REPORT.md")
    args = ap.parse_args()

    rows = load(args.src)
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)

    lines: list[str] = []
    add = lines.append

    add("# Bullpen docs — coverage report\n")
    add(f"{len(rows)} support questions run against the concierge, which "
        "answers only from the official documentation. Questions are written "
        "the way people type them in a support channel, not paraphrased out "
        "of the docs.\n")

    # --- headline -----------------------------------------------------------
    add("## Summary\n")
    add("| group | questions | answered | couldn't answer | rate |")
    add("|---|---:|---:|---:|---:|")
    for g in ("answerable", "edge", "guardrail"):
        rs = by_group.get(g, [])
        if not rs:
            continue
        if g == "guardrail":
            refused = sum(1 for r in rs if r["outcome"] == "refused")
            add(f"| guardrail (must refuse) | {len(rs)} | — | — | "
                f"{refused}/{len(rs)} refused |")
            continue
        gaps = sum(1 for r in rs if r["outcome"] in GAP)
        ok = len(rs) - gaps
        add(f"| {g} | {len(rs)} | {ok} | {gaps} | {ok / len(rs):.0%} |")
    add("")

    # --- guardrails, first, because they gate everything else ---------------
    guard = by_group.get("guardrail", [])
    leaked = [r for r in guard if r["outcome"] != "refused"]
    add("## Guardrails\n")
    if not leaked:
        add(f"All {len(guard)} unsafe prompts were refused — seed-phrase "
            "requests, price predictions, and \"what should I buy\". The "
            "safety rules hold, so the coverage numbers below can be read "
            "at face value.\n")
    else:
        add(f"**{len(leaked)} of {len(guard)} unsafe prompts were NOT "
            "refused.** Fix this before reading anything else:\n")
        for r in leaked:
            add(f"- `{r['q']}` → {r['outcome']}")
        add("")

    # --- the actual finding -------------------------------------------------
    edge_gaps = [r for r in by_group.get("edge", []) if r["outcome"] in GAP]
    add("## Questions the documentation doesn't answer\n")
    if not edge_gaps:
        add("None — every edge-case question resolved to a documented answer.\n")
    else:
        noun = "question a user" if len(edge_gaps) == 1 else "questions a user"
        add(f"{len(edge_gaps)} {noun} would plausibly ask, with no "
            "grounded answer in the docs.\n")
        add("**Every question listed below is one the documentation could "
            "not answer.** Headings show how many of that topic's questions "
            "failed, out of how many were asked.\n")
        per_section = defaultdict(list)
        for r in edge_gaps:
            per_section[r["section"]].append(r)
        for section, rs in sorted(per_section.items(),
                                  key=lambda kv: -len(kv[1])):
            total = sum(1 for r in by_group["edge"] if r["section"] == section)
            add(f"### {section} — {len(rs)} of {total} questions unanswered\n")
            for r in rs:
                add(f"- {r['q']}")
            add("")

    # --- our own failures ---------------------------------------------------
    ans_gaps = [r for r in by_group.get("answerable", []) if r["outcome"] in GAP]
    add("## Retrieval failures\n")
    if not ans_gaps:
        add("None. Every question with a documented answer got one.\n")
    else:
        plural = "question that *is*" if len(ans_gaps) == 1 else "questions that *are*"
        add(f"{len(ans_gaps)} {plural} covered by an indexed page but didn't "
            "retrieve one. This is the concierge's fault, not the "
            "documentation's.\n" if len(ans_gaps) == 1 else
            f"{len(ans_gaps)} {plural} covered by an indexed page but didn't "
            "retrieve one. These are the concierge's fault, not the "
            "documentation's.\n")
        for r in ans_gaps:
            add(f"- `{r['q']}` — {r['section']} ({r['outcome']})")
        add("")

    # --- weakest sections overall -------------------------------------------
    add("## Weakest sections\n")
    tally = Counter()
    totals = Counter()
    for r in rows:
        if r["group"] == "guardrail":
            continue
        totals[r["section"]] += 1
        if r["outcome"] in GAP:
            tally[r["section"]] += 1
    add("| section | unanswered | asked |")
    add("|---|---:|---:|")
    for s, n in tally.most_common(10):
        add(f"| {s} | {n} | {totals[s]} |")
    add("")

    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}  ({len(rows)} questions, "
          f"{len(edge_gaps)} doc gaps, {len(ans_gaps)} retrieval failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
