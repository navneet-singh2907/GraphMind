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
from src.retrieval.hybrid_langchain import answer_hybrid_langgraph
from src.retrieval.vector_rag import answer_question_vector


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
GRAPH_SEED_PATH = PROCESSED_DIR / "graph_seed.json"
VECTOR_DOCS_PATH = PROCESSED_DIR / "vector_documents.jsonl"


mcp = FastMCP(
    "GraphMind",
    instructions=(
        "GraphMind answers questions over a connected document knowledge base. "
        "Use ask_graphmind for normal questions, search_knowledge_base for "
        "source-grounded explanations, query_knowledge_graph for structured "
        "relationship lookups, and knowledge_stats for inventory."
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

    This hybrid tool automatically routes to Neo4j GraphRAG for structured
    relationship questions or Chroma/Nebius Vector RAG for document-backed
    explanations and summaries.
    """
    result = _quiet_call(answer_hybrid_langgraph, question)
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
    if first_word not in {"MATCH", "WITH", "RETURN"}:
        return {"error": "Only read-only Cypher queries are allowed."}

    blocked = [" CREATE ", " MERGE ", " DELETE ", " SET ", " REMOVE ", " DROP ", " LOAD CSV "]
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

    return {
        "documents": labels.get("Document", 0),
        "chunks": labels.get("Chunk", 0),
        "vector_documents": _count_jsonl(VECTOR_DOCS_PATH),
        "graph_nodes": len(seed.get("nodes", [])),
        "graph_relationships": len(seed.get("relationships", [])),
        "node_labels": dict(labels),
        "relationship_types": dict(rels),
    }


@mcp.resource("graphmind://stats")
def knowledge_stats_resource() -> str:
    """Current GraphMind inventory as JSON."""
    return json.dumps(knowledge_stats(), indent=2)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
