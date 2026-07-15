"""
Evaluation script — runs 10 test questions through both pipelines and
writes a markdown report to data/evaluation_report.md.

Usage:
    python scripts/evaluate.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.graph_rag import answer_question
from src.retrieval.vector_rag import answer_question_vector

TEST_QUESTIONS = [
    # (question, expected_pipeline_strength, category)
    ("Which concepts are connected to GraphRAG?",       "graph",  "entity lookup"),
    ("What tools does GraphMind use?",                  "graph",  "relationship traversal"),
    ("Which resources discuss knowledge graphs?",       "graph",  "relationship traversal"),
    ("How are semantic and graph retrieval connected?", "graph",  "structural"),
    ("Which concepts relate to vector search?",         "graph",  "structural"),
    ("What is retrieval-augmented generation?",         "vector", "concept definition"),
    ("How does GraphRAG differ from vector RAG?",       "both",   "content recall"),
    ("How does chunking affect RAG performance?",       "vector", "concept explanation"),
    ("How does hybrid retrieval choose a route?",       "both",   "architecture detail"),
    ("What tools support semantic retrieval?",          "graph",  "multi-hop"),
]

OUTPUT_PATH = os.path.join("data", "evaluation_report.md")


def run_eval():
    results = []
    total = len(TEST_QUESTIONS)

    print(f"Running {total} questions through both pipelines...\n")

    for i, (question, expected_strength, category) in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{total}] {question}")

        t0 = time.time()
        graph_result = answer_question(question)
        graph_latency = round(time.time() - t0, 2)

        t0 = time.time()
        vector_result = answer_question_vector(question)
        vector_latency = round(time.time() - t0, 2)

        results.append({
            "question":          question,
            "category":          category,
            "expected_strength": expected_strength,
            "graph": {
                "answer":   graph_result["answer"],
                "cypher":   graph_result["cypher"],
                "rows":     len(graph_result.get("graph_rows", [])),
                "nodes":    len(graph_result.get("viz_nodes", [])),
                "rels":     len(graph_result.get("viz_rels", [])),
                "related_content": graph_result.get("related_content", []),
                "latency_s": graph_latency,
            },
            "vector": {
                "answer":  vector_result["answer"],
                "sources": len(vector_result.get("sources", [])),
                "latency_s": vector_latency,
            },
        })

        print(f"  GraphRAG  ({graph_latency}s): {graph_result['answer'][:120].strip()}...")
        print(f"  VectorRAG ({vector_latency}s): {vector_result['answer'][:120].strip()}...")
        print()

    _write_report(results)
    print(f"\nEvaluation report written to {OUTPUT_PATH}")
    return results


def _write_report(results: list):
    lines = [
        "# GraphMind — Evaluation Report",
        "",
        "**System**: GraphRAG (Neo4j Cypher) vs Vector RAG (ChromaDB semantic search)  ",
        "**LLM**: Nemotron-3-Ultra-550B (Nebius Token Factory)  ",
        "**Embeddings**: Qwen3-Embedding-8B  ",
        "**Dataset**: the currently indexed GraphMind documents  ",
        "",
        "---",
        "",
        "## Summary table",
        "",
        "| # | Question | Category | Expected strength | GraphRAG rows | Vector sources | GraphRAG (s) | VectorRAG (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for i, r in enumerate(results, 1):
        q = r["question"][:55]
        lines.append(
            f"| {i} | {q} | {r['category']} | {r['expected_strength']} "
            f"| {r['graph']['rows']} rows / {r['graph']['nodes']} nodes "
            f"| {r['vector']['sources']} "
            f"| {r['graph']['latency_s']} "
            f"| {r['vector']['latency_s']} |"
        )

    avg_graph = round(sum(r["graph"]["latency_s"] for r in results) / len(results), 2)
    avg_vector = round(sum(r["vector"]["latency_s"] for r in results) / len(results), 2)
    lines += [
        "",
        f"**Average latency — GraphRAG**: {avg_graph}s  ",
        f"**Average latency — Vector RAG**: {avg_vector}s  ",
        "",
        "---",
        "",
        "## Per-question results",
        "",
    ]

    for i, r in enumerate(results, 1):
        lines += [
            f"### Q{i}: {r['question']}",
            "",
            f"**Category**: {r['category']}  ",
            f"**Expected pipeline strength**: {r['expected_strength']}",
            "",
            "#### GraphRAG",
            f"- Latency: {r['graph']['latency_s']}s",
            f"- Graph rows returned: {r['graph']['rows']}",
            f"- Visualization: {r['graph']['nodes']} nodes · {r['graph']['rels']} relationships",
        ]
        if r["graph"]["related_content"]:
            lines.append(f"- Related source chunks: {len(r['graph']['related_content'])}")
        lines += [
            f"- Cypher: `{r['graph']['cypher'][:200].strip()}`",
            "",
            "**Answer:**",
            "",
            r["graph"]["answer"],
            "",
            "#### Vector RAG",
            f"- Latency: {r['vector']['latency_s']}s",
            f"- Sources retrieved: {r['vector']['sources']}",
            "",
            "**Answer:**",
            "",
            r["vector"]["answer"],
            "",
            "---",
            "",
        ]

    lines += [
        "## Analysis",
        "",
        "### GraphRAG strengths",
        "- Precise entity lookups (documents, concepts, tools, and organizations) via structured Cypher",
        "- Multi-hop relationship traversal (document → chunk → concepts)",
        "- Deterministic answers for known entity relationships",
        "- Graph visualization shows how concepts connect",
        "- Related source chunk surfacing via DISCUSSES relationships",
        "",
        "### GraphRAG weaknesses",
        "- LLM-generated Cypher can hallucinate relationship directions or missing nodes",
        "- Answers only as good as graph extraction quality",
        "- Slow on cold start (Nebius API round-trip for Cypher + answer)",
        "",
        "### Vector RAG strengths",
        "- Strong on open-ended concept explanations from prose content",
        "- Source citations let users verify claims",
        "- Handles paraphrased or partial questions well",
        "- No schema dependency — works on any text",
        "",
        "### Vector RAG weaknesses",
        "- Weak on structural/relational queries (who is from which company)",
        "- Top-k retrieval can miss relevant segments when the query is too broad",
        "- Answers can be verbose without the conciseness of a graph result",
        "",
        "### Complementarity",
        "The two pipelines complement each other: GraphRAG wins on relational/structural ",
        "questions where exact entity traversal is needed; Vector RAG wins on conceptual ",
        "explanations where the answer lives in prose. Showing them side-by-side gives ",
        "users both the precise graph fact and the in-context explanation.",
    ]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_eval()
