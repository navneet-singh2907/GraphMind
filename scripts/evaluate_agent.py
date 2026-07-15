"""Dataset-independent evaluation harness for the GraphMind retrieval agent."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.orchestrator import answer_agentic


def evaluate_case(case: dict) -> dict:
    started = time.perf_counter()
    result = answer_agentic(case["question"], collection=case.get("collection"))
    latency = round(time.perf_counter() - started, 3)
    answer_lower = result["answer"].lower()
    expected_terms = [str(term).lower() for term in case.get("expected_terms", [])]
    matched_terms = [term for term in expected_terms if term in answer_lower]
    return {
        "question": case["question"],
        "collection": case.get("collection"),
        "answer": result["answer"],
        "latency_seconds": latency,
        "attempts": result["attempts"],
        "evidence_count": result["evidence_count"],
        "source_count": len(result["sources"]),
        "confidence": result["verification"]["confidence"],
        "sufficient": result["verification"]["sufficient"],
        "expected_terms": expected_terms,
        "matched_terms": matched_terms,
        "term_recall": round(len(matched_terms) / len(expected_terms), 3) if expected_terms else None,
        "plan": result["plan"],
        "trace": result["trace"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GraphMind on a portable question set.")
    parser.add_argument("--dataset", default="evaluation/questions.example.json")
    parser.add_argument("--output", default="data/evaluation/agent_report.json")
    args = parser.parse_args()

    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    results = [evaluate_case(case) for case in cases]
    scored = [row["term_recall"] for row in results if row["term_recall"] is not None]
    report = {
        "summary": {
            "questions": len(results),
            "sufficient_rate": round(sum(row["sufficient"] for row in results) / len(results), 3) if results else 0,
            "grounded_rate": round(sum(row["source_count"] > 0 for row in results) / len(results), 3) if results else 0,
            "average_confidence": round(sum(row["confidence"] for row in results) / len(results), 3) if results else 0,
            "average_term_recall": round(sum(scored) / len(scored), 3) if scored else None,
            "average_latency_seconds": round(sum(row["latency_seconds"] for row in results) / len(results), 3) if results else 0,
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote evaluation report to {output}")


if __name__ == "__main__":
    main()
