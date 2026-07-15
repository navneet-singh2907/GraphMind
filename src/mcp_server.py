"""
GraphMind MCP server.

Run with:
    python -m src.mcp_server

The server talks over stdio by default, which is what most MCP clients expect.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from src.retrieval.graph_rag import answer_question as answer_graph
from src.retrieval.graph_rag import run_cypher
from src.agent.orchestrator import answer_agentic
from src.retrieval.tools import inspect_source as inspect_source_tool
from src.retrieval.tools import keyword_search as keyword_search_tool
from src.retrieval.tools import metadata_search as metadata_search_tool
from src.retrieval.vector_rag import answer_question_vector


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
GRAPH_SEED_PATH = PROCESSED_DIR / "graph_seed.json"
VECTOR_DOCS_PATH = PROCESSED_DIR / "vector_documents.jsonl"
INGESTION_MANIFEST_PATH = PROCESSED_DIR / "ingestion_manifest.json"


mcp = FastMCP(
    "GraphMind",
    instructions=(
        "GraphMind answers questions over a connected document knowledge base. "
        "Use ask_graphmind for planned, verified multi-tool retrieval; "
        "search_knowledge_base or search_keywords for direct retrieval; "
        "query_knowledge_graph for structured relationships; search_metadata "
        "and inspect_source for source discovery; and knowledge_stats for inventory."
    ),
)


def _quiet_call(fn: Callable[..., dict], *args: Any, **kwargs: Any) -> dict:
    """Prevent retriever debug prints from corrupting MCP stdio."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _summarize_result(result: dict) -> dict:
    return {
        "question": result.get("question"),
        "answer": result.get("answer"),
        "route": result.get("route"),
        "route_label": result.get("route_label"),
        "route_reasoning": result.get("route_reasoning"),
        "fallback_used": result.get("fallback_used"),
        "cypher": result.get("cypher"),
        "graph_rows": result.get("graph_rows", []),
        "sources": result.get("sources", []),
        "related_content": result.get("related_content", []),
        "plan": result.get("plan"),
        "verification": result.get("verification"),
        "attempts": result.get("attempts"),
        "trace": result.get("trace", []),
    }


def _load_graph_seed() -> dict:
    if not GRAPH_SEED_PATH.exists():
        return {"nodes": [], "relationships": []}
    return json.loads(GRAPH_SEED_PATH.read_text(encoding="utf-8"))


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


@mcp.tool()
def ask_graphmind(question: str) -> dict:
    """
    Ask GraphMind a question about the indexed knowledge base.

    The bounded retrieval agent plans tool calls, combines semantic, lexical,
    metadata, source, and graph evidence, verifies sufficiency, and retries once
    before producing a cited answer.
    """
    result = _quiet_call(answer_agentic, question)
    return _summarize_result(result)


@mcp.tool()
def search_knowledge_base(question: str) -> dict:
    """
    Search indexed documents and notes semantically.

    Best for explanations, summaries, quotations, and source-backed detail.
    """
    result = _quiet_call(answer_question_vector, question)
    return _summarize_result(result)


@mcp.tool()
def search_keywords(query: str, collection: str | None = None, limit: int = 8) -> dict:
    """Run exact-term BM25 retrieval over canonical document chunks."""
    evidence = keyword_search_tool(query=query, collection=collection, k=limit)
    return {"results": [item.to_dict() for item in evidence], "result_count": len(evidence)}


@mcp.tool()
def search_metadata(
    query: str = "", collection: str | None = None, source_type: str | None = None, limit: int = 20
) -> dict:
    """Find indexed content by title, collection, source type, or parser metadata."""
    evidence = metadata_search_tool(
        query=query, collection=collection, source_type=source_type, k=limit
    )
    return {"results": [item.to_dict() for item in evidence], "result_count": len(evidence)}


@mcp.tool()
def inspect_source(source_uri: str, limit: int = 50) -> dict:
    """Read ordered chunks from one indexed source URI."""
    evidence = inspect_source_tool(source_uri=source_uri, k=limit)
    return {"results": [item.to_dict() for item in evidence], "result_count": len(evidence)}


@mcp.tool()
def query_knowledge_graph(question: str) -> dict:
    """
    Query the Neo4j knowledge graph.

    Best for documents, people, organizations, concepts, tools, projects,
    resources, entity relationships, and other structured lookups.
    """
    result = _quiet_call(answer_graph, question)
    return _summarize_result(result)


@mcp.tool()
def run_readonly_cypher(cypher: str) -> dict:
    """
    Run a read-only Cypher query against the GraphMind Neo4j graph.

    Only MATCH/RETURN style queries are allowed. Use this when the LLM needs
    precise graph rows beyond the natural-language graph tool.
    """
    first_word = cypher.strip().split(maxsplit=1)[0].upper() if cypher.strip() else ""
    if first_word not in {"MATCH", "OPTIONAL", "WITH", "RETURN", "UNWIND"}:
        return {"error": "Only read-only Cypher queries are allowed."}

    blocked = [
        " CREATE ", " MERGE ", " DELETE ", " SET ", " REMOVE ", " DROP ",
        " LOAD CSV ", " DETACH ", " CALL ", " FOREACH ",
    ]
    padded = f" {cypher.upper()} "
    if any(token in padded for token in blocked):
        return {"error": "Mutation Cypher is not allowed from the MCP tool."}

    rows = run_cypher(cypher)
    return {"rows": rows, "row_count": len(rows)}


@mcp.tool()
def knowledge_stats() -> dict:
    """
    Return the current GraphMind inventory from processed data.
    """
    seed = _load_graph_seed()
    labels = Counter(node.get("label", "Unknown") for node in seed.get("nodes", []))
    rels = Counter(rel.get("type", "UNKNOWN") for rel in seed.get("relationships", []))

    manifest = {}
    if INGESTION_MANIFEST_PATH.exists():
        manifest = json.loads(INGESTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    collections = sorted({item.get("collection", "default") for item in manifest.values()})
    source_types = sorted({Path(source).suffix.lower().lstrip(".") for source in manifest})

    return {
        "documents": labels.get("Document", 0),
        "chunks": labels.get("Chunk", 0),
        "vector_documents": _count_jsonl(VECTOR_DOCS_PATH),
        "graph_nodes": len(seed.get("nodes", [])),
        "graph_relationships": len(seed.get("relationships", [])),
        "node_labels": dict(labels),
        "relationship_types": dict(rels),
        "collections": collections,
        "source_types": source_types,
    }


@mcp.resource("graphmind://stats")
def knowledge_stats_resource() -> str:
    """Current GraphMind inventory as JSON."""
    return json.dumps(knowledge_stats(), indent=2)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
