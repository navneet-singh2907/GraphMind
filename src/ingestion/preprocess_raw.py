"""Generic document preprocessing for GraphMind."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".json", ".jsonl", ".yaml", ".yml",
}


def normalize_text(text: str) -> str:
    """Normalize newlines and repeated whitespace without changing meaning."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _chunk_text(text: str, target_chars: int = 1800, overlap_chars: int = 200) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if buffer and len(candidate) > target_chars:
            chunks.append(buffer)
            overlap = buffer[-overlap_chars:].lstrip() if overlap_chars else ""
            buffer = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph
        else:
            buffer = candidate

    if buffer:
        chunks.append(buffer)
    return chunks


def _stable_id(relative_path: str, chunk_index: int) -> str:
    digest = hashlib.sha1(f"{relative_path}:{chunk_index}".encode("utf-8")).hexdigest()[:16]
    return f"chunk_{digest}"


def preprocess_document(path: Path, raw_dir: Path = RAW_DIR) -> list[dict]:
    relative_path = path.relative_to(raw_dir).as_posix()
    text = normalize_text(_read_document(path))
    title = path.stem.replace("_", " ").replace("-", " ").strip()
    collection = path.parent.relative_to(raw_dir).as_posix()
    if collection == ".":
        collection = "default"

    records = []
    for index, chunk in enumerate(_chunk_text(text)):
        records.append(
            {
                "id": _stable_id(relative_path, index),
                "document_id": hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16],
                "title": title,
                "source_path": relative_path,
                "source_type": path.suffix.lower().lstrip(".") or "text",
                "collection": collection,
                "chunk_index": index,
                "text": chunk,
            }
        )
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def preprocess_all(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> dict[str, int]:
    """Convert all supported source files into generic document chunks."""
    supported = SUPPORTED_TEXT_EXTENSIONS | {".pdf"}
    paths = sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported and not path.name.startswith(".")
    ) if raw_dir.exists() else []

    records: list[dict] = []
    for path in paths:
        records.extend(preprocess_document(path, raw_dir))

    document_count = write_jsonl(processed_dir / "document_chunks.jsonl", records)
    vector_count = write_jsonl(processed_dir / "vector_documents.jsonl", records)
    return {
        "documents": len(paths),
        "document_chunks": document_count,
        "vector_documents": vector_count,
    }
