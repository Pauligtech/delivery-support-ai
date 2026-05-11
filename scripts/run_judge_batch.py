"""
scripts/run_judge_batch.py
─────────────────────────────────────────────────────────────────────────────
Batch LLM Judge Runner
Reads all logged interactions from logs/interaction_log.jsonl,
runs the Judge on each one, and prints a quality report.

Usage:
    python scripts/run_judge_batch.py
    python scripts/run_judge_batch.py --limit 50   # evaluate last 50 only
─────────────────────────────────────────────────────────────────────────────
"""

import json
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path so core.* imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.judge import run_judge, QUALITY_THRESHOLDS
from core.policy import get_policy_version

logging.basicConfig(level=logging.WARNING)

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "interaction_log.jsonl"


def load_interactions(limit: int | None = None) -> list[dict]:
    if not LOG_FILE.exists():
        print(f"❌ Log file not found: {LOG_FILE}")
        print("   Start the API and have some conversations first.")
        sys.exit(1)

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if limit:
        lines = lines[-limit:]  # Evaluate most recent N

    interactions = []
    for line in lines:
        try:
            interactions.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            continue
    return interactions


def run_batch(limit: int | None = None) -> None:
    interactions = load_interactions(limit)

    if not interactions:
        print("No interactions found in the log file.")
        return

    print(f"\n{'='*65}")
    print(f"  📊 DoorDash Dasher Support — LLM Judge Batch Report")
    print(f"  Policy Version : {get_policy_version()}")
    print(f"  Interactions   : {len(interactions)}")
    print(f"{'='*65}\n")

    results = []
    policy_sync_issues = 0
    verdicts = {"YES": 0, "NO": 0, "ERROR": 0}

    for i, entry in enumerate(interactions, 1):
        query    = entry.get("query", "")
        context  = entry.get("context_snippet", "")
        response = entry.get("raw_response", "")
        guardrail_result = entry.get("guardrail", {}) or {"verdict": "YES", "policy_version_checked": "unknown"}

        print(f"  [{i:03d}/{len(interactions):03d}] Evaluating: {query[:60]}...")

        result = run_judge(
            query=query,
            context=context,
            response=response,
            guardrail_result=guardrail_result,
        )

        results.append(result)
        verdict = result.get("verdict", "ERROR")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1

        if result.get("policy_sync_issue"):
            policy_sync_issues += 1

    # ── Summary Report ────────────────────────────────────────────────────
    def avg(key: str) -> float:
        vals = [r.get("scores", {}).get(key, 0) for r in results if "scores" in r]
        return sum(vals) / len(vals) if vals else 0.0

    overall_avg = sum(r.get("overall_score", 0) for r in results) / len(results)

    print(f"\n{'─'*65}")
    print("  VERDICT SUMMARY")
    print(f"{'─'*65}")
    print(f"  ✅ YES (Delivered)      : {verdicts.get('YES', 0)}")
    print(f"  ❌ NO  (Blocked)        : {verdicts.get('NO', 0)}")
    print(f"  ⚠️  Errors              : {verdicts.get('ERROR', 0)}")
    print(f"  🔄 Policy Sync Issues   : {policy_sync_issues}")

    print(f"\n{'─'*65}")
    print("  QUALITY SCORES (average /5 | min threshold)")
    print(f"{'─'*65}")

    metrics = [
        ("retrieval_correctness", "Retrieval Correctness"),
        ("response_accuracy",     "Response Accuracy    "),
        ("grammar_language",      "Grammar & Language   "),
        ("coherence_to_context",  "Coherence to Context "),
        ("relevance_to_request",  "Relevance to Request "),
    ]

    for key, label in metrics:
        score     = avg(key)
        threshold = QUALITY_THRESHOLDS[key]
        status    = "✅" if score >= threshold else "⚠️ BELOW THRESHOLD"
        bar_filled = int(score / 5 * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"  {label} : {score:.2f}  [{bar}]  min={threshold}  {status}")

    print(f"\n  Overall Score          : {overall_avg:.2f} / 5")
    print(f"\n{'='*65}\n")

    # ── Save detailed results ────────────────────────────────────────────
    output_path = LOG_FILE.parent / "judge_report.json"
    report = {
        "policy_version": get_policy_version(),
        "total_interactions": len(interactions),
        "verdicts": verdicts,
        "policy_sync_issues": policy_sync_issues,
        "averages": {key: round(avg(key), 2) for key, _ in metrics},
        "overall_score": round(overall_avg, 2),
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  📁 Detailed report saved to: {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Judge on logged interactions")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate last N interactions")
    args = parser.parse_args()
    run_batch(limit=args.limit)
