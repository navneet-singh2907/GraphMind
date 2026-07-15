import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.preprocess_raw import preprocess_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally ingest GraphMind source documents.")
    parser.add_argument("--collection", help="Override the collection name for all discovered files.")
    parser.add_argument("--force", action="store_true", help="Reparse unchanged files.")
    args = parser.parse_args()
    counts = preprocess_all(collection=args.collection, force=args.force)
    print("Preprocessing complete:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
