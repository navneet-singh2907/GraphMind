"""Retrieval tools exposed to the GraphMind agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from src.retrieval.keyword_search import KeywordIndex
from src.retrieval.models import Evidence


CHUNKS_PATH = Path("data/processed/document_chunks.jsonl")


def _load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        return []
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def vector_search(query: str, k: int = 8, collection: str | None = None, **_: Any) -> list[Evidence]:
    from src.retrieval.vector_rag import semantic_search

    filters = {"collection": collection} if collection else None
    return [
        Evidence(
            tool="vector_search",
            content=document.page_content,
            source_uri=document.metadata.get("source") or document.metadata.get("source_uri", ""),
            title=document.metadata.get("title", ""),
            metadata=dict(document.metadata),
        )
        for document in semantic_search(query, k=k, filters=filters)
    ]


def keyword_search(
    query: str,
    k: int = 8,
    collection: str | None = None,
    source_type: str | None = None,
    **_: Any,
) -> list[Evidence]:
    return [
        Evidence(
            tool="keyword_search",
            content=row["content"],
            source_uri=row["source_uri"],
            title=row["title"],
            score=row["score"],
            metadata={key: value for key, value in row.items() if key not in {"content"}},
        )
        for row in KeywordIndex.from_jsonl().search(
            query, k=k, collection=collection, source_type=source_type
        )
    ]


def metadata_search(
    query: str = "",
    k: int = 20,
    collection: str | None = None,
    source_type: str | None = None,
    **_: Any,
) -> list[Evidence]:
    lowered = query.lower().strip()
    matches = []
    for row in _load_chunks():
        if collection and row.get("collection") != collection:
            continue
        if source_type and row.get("source_type") != source_type:
            continue
        haystack = " ".join(
            [row.get("title", ""), row.get("source_uri", ""), json.dumps(row.get("metadata", {}))]
        ).lower()
        if lowered and lowered not in haystack:
            continue
        matches.append(
            Evidence(
                tool="metadata_search",
                content=row["content"],
                source_uri=row["source_uri"],
                title=row["title"],
                metadata={key: value for key, value in row.items() if key != "content"},
            )
        )
        if len(matches) >= k:
            break
    return matches


def inspect_source(source_uri: str, k: int = 50, **_: Any) -> list[Evidence]:
    rows = [row for row in _load_chunks() if row.get("source_uri") == source_uri]
    rows.sort(key=lambda row: (row.get("document_id", ""), row.get("chunk_index", 0)))
    return [
        Evidence(
            tool="inspect_source",
            content=row["content"],
            source_uri=row["source_uri"],
            title=row["title"],
            metadata={key: value for key, value in row.items() if key != "content"},
        )
        for row in rows[:k]
    ]


def graph_search(query: str, k: int = 20, **_: Any) -> list[Evidence]:
    from src.retrieval.graph_rag import query_graph

    cypher, rows = query_graph(query)
    rows = rows[:k]
    return [
        Evidence(
            tool="graph_search",
            content=json.dumps(row, ensure_ascii=False, default=str),
            metadata={"cypher": cypher, "row": row},
        )
        for row in rows
    ]


TOOL_REGISTRY: dict[str, Callable[..., list[Evidence]]] = {
    "vector_search": vector_search,
    "keyword_search": keyword_search,
    "metadata_search": metadata_search,
    "inspect_source": inspect_source,
    "graph_search": graph_search,
}

TOOL_DESCRIPTIONS = {
    "vector_search": "Semantic similarity over document chunks; best for paraphrases and concepts.",
    "keyword_search": "BM25 lexical search; best for exact names, identifiers, and quoted language.",
    "metadata_search": "Filter or find documents by title, collection, source type, or parser metadata.",
    "inspect_source": "Read ordered chunks from one known source URI.",
    "graph_search": "Traverse entities and relationships in Neo4j for structured or multi-hop questions.",
}


def execute_tool(name: str, arguments: dict[str, Any]) -> list[Evidence]:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown retrieval tool: {name}")
    return TOOL_REGISTRY[name](**arguments)
