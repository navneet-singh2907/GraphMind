"""Canonical ingestion models shared by every GraphMind parser."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class ParsedDocument:
    """Parser output before chunking."""

    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentChunk:
    """Data-agnostic contract consumed by all GraphMind indexes."""

    id: str
    document_id: str
    title: str
    content: str
    source_uri: str
    source_type: str
    collection: str
    content_hash: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DocumentChunk":
        return cls(
            id=record["id"],
            document_id=record["document_id"],
            title=record["title"],
            content=record.get("content") or record.get("text", ""),
            source_uri=record.get("source_uri") or record.get("source_path", ""),
            source_type=record.get("source_type", "text"),
            collection=record.get("collection", "default"),
            content_hash=record.get("content_hash", ""),
            chunk_index=int(record.get("chunk_index", 0)),
            metadata=dict(record.get("metadata", {})),
            schema_version=record.get("schema_version", SCHEMA_VERSION),
        )
