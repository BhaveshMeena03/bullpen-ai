"""Measure whether answers stay inside their sources, on both surfaces.

    python evals/run_faithfulness.py [--limit N] [--search-only|--chat-only]

Every other harness here grades an answer by reading the answer. This one
reads the answer next to the material it was built from, which is the only
way to catch the failure that matters most: a specific, confident claim
attributed to a real named person who never made it.

Two passes per question:

  deterministic   quotes, timestamps and figures, checked by string
                  containment against the retrieved windows (faithfulness.py)
  judged          a model shown BOTH the sources and the answer, asked
                  whether every specific claim traces back

The deterministic pass is the one to trust. A quote that appears in no
retrieved window is a fact about the output, not an opinion about it, and it
stays true whichever model grades the run.

Writes evals/FAITHFULNESS_REPORT.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
from anthropic import Anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.faithfulness import audit, build_faithful_request  # noqa: E402
from evals.judge import parse  # noqa: E402
from evals.questions import ANSWERABLE  # noqa: E402
from evals.search_questions import BROAD, TARGETED, TIME  # noqa: E402

SEARCH_BASE = "https://search.lexthedev.com"
CHAT_BASE = "https://concierge.lexthedev.com"
HERE = Path(__file__).resolve().parent
DELAY = 1.5
JUDGE_MODEL = "claude-haiku-4-5"


def post(client, url, payload, attempts=4):
    """Retry the way run_search_eval does: a free-tier host dropping one
    connection must not discard a run that is most of the way through."""
    last = None
    for attempt in range(attempts):
        try:
            r = client.post(url, json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < attempts - 1:
                time.sleep(2 ** attempt * 2)
    print(f"    ! giving up: {type(last).__name__}")
    return None


def ask_search(client, base, question, top_k):
    d = post(client, f"{base}/v1/podcast/search",
             {"query": question, "top_k": top_k})
    if not d:
        return None
    return d.get("answer", ""), d.get("hits", [])


def ask_chat(client, base, question):
    d = post(client, f"{base}/v1/chat", {"message": question})
    if not d:
        return None
    # RetrievedChunk keeps the human-readable name under metadata, and has no
    # timestamps — the timestamp check simply finds nothing to check, which is
    # correct rather than a gap.
    sources = [{"text": s.get("text", ""),
                "title": (s.get("metadata") or {}).get("title")}
               for s in d.get("sources", [])]
    return d.get("answer", ""), sources


def judge_one(judge, question, answer, sources):
    msg = judge.messages.create(
        **build_faithful_request(JUDGE_MODEL, question, answer, sources))
    return parse("".join(b.text for b in msg.content if b.type == "text"))


def run_surface(name, questions, ask, judge, raw):
    """One surface, both passes. Returns (hard, soft, judged_fail, total)."""
    print(f"\n{name}: {len(questions)} questions")
    hard = soft = judged_fail = 0
    detail = []
    for question in questions:
        got = ask(question)
        if got is None:
            continue
        answer, sources = got
        if not answer.strip():
            print(f"  ---- {question[:56]} (no answer)")
            continue

        findings = audit(answer, sources)
        hard_here = [f for f in findings if f.hard]
        soft_here = [f for f in findings if not f.hard]
        ok, why = judge_one(judge, question, answer, sources)

        hard += len(hard_here)
        soft += len(soft_here)
        judged_fail += (not ok)

        mark = "FAB " if hard_here else ("  ? " if (soft_here or not ok) else "  ok")
        print(f"  {mark} {question[:56]}")
        for f in hard_here:
            print(f"        {f}")

        detail.append((question, hard_here, soft_here, ok, why))
        raw.append({"surface": name, "q": question, "answer": answer,
                    "n_sources": len(sources),
                    "hard": [str(f) for f in hard_here],
                    "soft": [str(f) for f in soft_here],
                    "judged_ok": ok, "judge_why": why})
        time.sleep(DELAY)
    return hard, soft, judged_fail, detail


def section(add, title, detail):
    add(f"### {title}\n")
    clean = True
    for question, hard, soft, ok, why in detail:
        if not hard and not soft and ok:
            continue
        clean = False
        add(f"- `{question}`")
        for f in hard:
            add(f"  - **{f}**")
        for f in soft:
            add(f"  - {f}")
        if not ok:
            add(f"  - judge: {why.strip()[:200]}")
    if clean:
        add("Nothing flagged.")
    add("")


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-base", default=SEARCH_BASE)
    ap.add_argument("--chat-base", default=CHAT_BASE)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--limit", type=int, default=12,
                    help="questions per surface; these are live paid calls")
    ap.add_argument("--search-only", action="store_true")
    ap.add_argument("--chat-only", action="store_true")
    args = ap.parse_args()

    judge = Anthropic()
    raw: list[dict] = []
    results = {}

    # Questions most likely to pull a specific claim out of the model:
    # targeted recall, broad opinion, and chronology all invite quoting.
    search_qs = ([q for q, _ in TARGETED] + BROAD + TIME)[:args.limit]
    # For the concierge, operational questions are the dangerous ones —
    # fees, thresholds, eligibility. Those are where a guess costs money.
    chat_qs = [q for group in ANSWERABLE.values() for q in group][:args.limit]

    with httpx.Client(timeout=180.0) as c:
        if not args.chat_only:
            results["search"] = run_surface(
                "market search", search_qs,
                lambda q: ask_search(c, args.search_base, q, args.top_k),
                judge, raw)
        if not args.search_only:
            results["chat"] = run_surface(
                "support bot", chat_qs,
                lambda q: ask_chat(c, args.chat_base, q),
                judge, raw)

    lines: list[str] = []
    add = lines.append
    add("# Faithfulness — do answers stay inside their sources?\n")
    add("Each answer is checked against the exact material it was built "
        "from. Quotes are verified by containment, which is a fact about "
        "the output; timestamps and figures are reported with context, "
        "because rounding and unit changes are legitimate.\n")

    add("| surface | questions | fabricated quotes | unverified | judged unsupported |")
    add("|---|---:|---:|---:|---:|")
    for key, label in (("search", "market search"), ("chat", "support bot")):
        if key not in results:
            continue
        hard, soft, jf, detail = results[key]
        add(f"| {label} | {len(detail)} | {hard} | {soft} | {jf} |")
    add("")
    add("A non-zero **fabricated quotes** column is a release blocker: it "
        "means the tool put words in someone's mouth. The other two columns "
        "are for reading, not for gating.\n")

    for key, label in (("search", "Market search"), ("chat", "Support bot")):
        if key in results:
            section(add, label, results[key][3])

    (HERE / "faithfulness_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in raw))
    out = HERE / "FAITHFULNESS_REPORT.md"
    out.write_text("\n".join(lines))
    print(f"\nwrote {out}")

    blocking = sum(results[k][0] for k in results)
    if blocking:
        print(f"FAIL: {blocking} fabricated quote(s)")
        return 1
    print("no fabricated quotes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
