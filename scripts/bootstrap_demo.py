"""Ingest the bundled synthetic corpus and optionally test keyword retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.preprocess_raw import preprocess_all
from src.retrieval.keyword_search import KeywordIndex


DEMO_DIR = ROOT / "demo_data" / "sources"
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the synthetic HelioDesk demo corpus.")
    parser.add_argument("--force", action="store_true", help="Reparse every demo file.")
    parser.add_argument(
        "--question",
        default="Audit Store owner Trust Team recovery_time_target_minutes",
        help="Keyword-retrieval smoke-test question.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of smoke-test results.")
    args = parser.parse_args()

    counts = preprocess_all(DEMO_DIR, PROCESSED_DIR, force=args.force)
    print("Demo ingestion complete:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    index = KeywordIndex.from_jsonl(PROCESSED_DIR / "document_chunks.jsonl")
    print(f"\nKeyword smoke test: {args.question}")
    for rank, result in enumerate(index.search(args.question, k=args.top_k), start=1):
        preview = result["content"].replace("\n", " ")[:180]
        print(f"  {rank}. {result['source_uri']} ({result['score']:.3f}) - {preview}")


if __name__ == "__main__":
    main()
