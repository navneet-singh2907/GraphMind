"""Prepare the bundled demo indexes and start GraphMind in a container."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROCESSED_CHUNKS = ROOT / "data" / "processed" / "document_chunks.jsonl"
CHROMA_READY = ROOT / "chroma_db" / ".graphmind-ready"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def validated_port() -> int:
    raw = os.getenv("PORT", "8501")
    try:
        port = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"PORT must be an integer, received {raw!r}") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError(f"PORT must be between 1 and 65535, received {port}")
    return port


def require_runtime_configuration() -> None:
    missing = [
        name
        for name in ("NEBIUS_API_KEY",)
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(
            "Missing required runtime configuration: "
            + ", ".join(missing)
            + ". Inject it through the container platform's secret manager."
        )


def prepare_demo_indexes() -> None:
    """Create ephemeral canonical and vector indexes from public synthetic data."""
    from src.ingestion.preprocess_raw import preprocess_all
    from src.retrieval.vector_rag import build_vector_store

    if not PROCESSED_CHUNKS.exists():
        print("GraphMind startup: processing the bundled synthetic demo corpus.", flush=True)
        counts = preprocess_all(
            raw_dir=ROOT / "demo_data" / "sources",
            processed_dir=ROOT / "data" / "processed",
            force=True,
        )
        print(f"GraphMind startup: ingestion complete: {counts}", flush=True)

    if not CHROMA_READY.exists():
        print("GraphMind startup: building the ephemeral Chroma demo index.", flush=True)
        build_vector_store()
        CHROMA_READY.parent.mkdir(parents=True, exist_ok=True)
        CHROMA_READY.write_text("ready\n", encoding="utf-8")
        print("GraphMind startup: vector index is ready.", flush=True)


def start_streamlit() -> None:
    port = validated_port()
    command = [
        "streamlit",
        "run",
        "src/ui/app.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    print(f"GraphMind startup: listening on 0.0.0.0:{port}.", flush=True)
    os.execvp(command[0], command)


def main() -> None:
    os.chdir(ROOT)
    if env_flag("GRAPHMIND_BOOTSTRAP_DEMO", default=True):
        require_runtime_configuration()
        prepare_demo_indexes()
    start_streamlit()


if __name__ == "__main__":
    main()
