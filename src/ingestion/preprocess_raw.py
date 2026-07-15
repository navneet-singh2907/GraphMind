"""Incremental, parser-driven ingestion pipeline for GraphMind."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from src.ingestion.models import DocumentChunk
from src.ingestion.parsers import ParserRegistry


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHUNKS_PATH = PROCESSED_DIR / "document_chunks.jsonl"
VECTOR_PATH = PROCESSED_DIR / "vector_documents.jsonl"
MANIFEST_PATH = PROCESSED_DIR / "ingestion_manifest.json"
ERRORS_PATH = PROCESSED_DIR / "ingestion_errors.json"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: Iterable[dict]) -> int:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _collection_for(path: Path, raw_dir: Path, override: str | None) -> str:
    if override:
        return override
    relative = path.relative_to(raw_dir)
    return relative.parts[0] if len(relative.parts) > 1 else "default"


def preprocess_document(
    path: Path,
    raw_dir: Path = RAW_DIR,
    registry: ParserRegistry | None = None,
    collection: str | None = None,
) -> list[dict]:
    registry = registry or ParserRegistry()
    source_uri = path.relative_to(raw_dir).as_posix()
    source_type = path.suffix.lower().lstrip(".") or "text"
    content_hash = _file_hash(path)
    resolved_collection = _collection_for(path, raw_dir, collection)
    records: list[dict] = []

    for part_index, parsed in enumerate(registry.parse(path)):
        document_id = _short_hash(f"{source_uri}:{part_index}")
        for chunk_index, content in enumerate(_chunk_text(normalize_text(parsed.content))):
            chunk = DocumentChunk(
                id=f"chunk_{_short_hash(f'{document_id}:{chunk_index}:{content}')}",
                document_id=document_id,
                title=parsed.title,
                content=content,
                source_uri=source_uri,
                source_type=source_type,
                collection=resolved_collection,
                content_hash=content_hash,
                chunk_index=chunk_index,
                metadata={"part_index": part_index, **parsed.metadata},
            )
            records.append(chunk.to_record())
    return records


def preprocess_all(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    collection: str | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Incrementally normalize every registered file into canonical chunks."""
    registry = ParserRegistry()
    chunk_path = processed_dir / CHUNKS_PATH.name
    vector_path = processed_dir / VECTOR_PATH.name
    manifest_path = processed_dir / MANIFEST_PATH.name
    errors_path = processed_dir / ERRORS_PATH.name

    previous_rows = _read_jsonl(chunk_path)
    previous_by_source: dict[str, list[dict]] = {}
    for row in previous_rows:
        source_uri = row.get("source_uri") or row.get("source_path", "")
        previous_by_source.setdefault(source_uri, []).append(row)

    paths = sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in registry.supported_extensions
    ) if raw_dir.exists() else []

    rows: list[dict] = []
    manifest: dict[str, dict] = {}
    errors: list[dict] = []
    parsed_count = 0
    skipped_count = 0

    for path in paths:
        source_uri = path.relative_to(raw_dir).as_posix()
        content_hash = _file_hash(path)
        cached = previous_by_source.get(source_uri, [])
        if not force and cached and all(row.get("content_hash") == content_hash for row in cached):
            rows.extend(cached)
            skipped_count += 1
            parser_name = cached[0].get("metadata", {}).get("parser", "cached")
        else:
            try:
                parsed_rows = preprocess_document(path, raw_dir, registry, collection)
                parser_name = registry.parser_for(path).__class__.__name__
                for row in parsed_rows:
                    row.setdefault("metadata", {})["parser"] = parser_name
                rows.extend(parsed_rows)
                parsed_count += 1
            except Exception as exc:
                errors.append({"source_uri": source_uri, "error": str(exc), "type": type(exc).__name__})
                continue

        manifest[source_uri] = {
            "content_hash": content_hash,
            "parser": parser_name,
            "collection": _collection_for(path, raw_dir, collection),
            "size_bytes": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }

    rows.sort(key=lambda row: (row["source_uri"], row["document_id"], row["chunk_index"]))
    chunk_count = _write_jsonl(chunk_path, rows)
    _write_jsonl(vector_path, rows)
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    errors_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "discovered_files": len(paths),
        "parsed_files": parsed_count,
        "unchanged_files": skipped_count,
        "failed_files": len(errors),
        "document_chunks": chunk_count,
        "removed_files": len(set(previous_by_source) - set(manifest)),
    }
