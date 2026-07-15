from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.preprocess_raw import preprocess_all


if __name__ == "__main__":
    counts = preprocess_all()
    print("Preprocessing complete:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
