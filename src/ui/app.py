"""Streamlit interface for GraphMind."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import streamlit_agraph as _sagraph
from streamlit_agraph import Config, Edge, Node

from src.retrieval.graph_rag import answer_question
from src.retrieval.hybrid_langchain import LANGGRAPH_AVAILABLE, answer_hybrid_langgraph
from src.retrieval.vector_rag import answer_question_vector


st.set_page_config(page_title="GraphMind", page_icon="🧠", layout="wide")

NODE_COLORS = {
    "Document": "#4f8ef7",
    "Chunk": "#a78bfa",
    "Person": "#f43f5e",
    "Organization": "#f59e0b",
    "Topic": "#06b6d4",
    "Concept": "#34d399",
    "Tool": "#f97316",
    "Project": "#6366f1",
    "Resource": "#94a3b8",
}

DEMO_QUESTIONS = [
    "Which concepts are connected to GraphRAG?",
    "Which documents discuss vector search?",
    "What tools does GraphMind use?",
    "How does hybrid retrieval work?",
    "Which resources reference knowledge graphs?",
    "Summarize the retrieval architecture.",
]

for key in ("hybrid_history", "graph_history", "vector_history"):
    st.session_state.setdefault(key, [])
st.session_state.setdefault("mode", "hybrid")


def render_graph(viz_nodes: list, viz_rels: list, key: str) -> None:
    if not viz_nodes:
        st.info("No connected graph data was returned for this question.")
        return
    nodes = [
        Node(
            id=node["id"],
            label=node.get("name", node["id"]),
            color=NODE_COLORS.get(node.get("label"), "#94a3b8"),
            size=22,
        )
        for node in viz_nodes
    ]
    edges = [
        Edge(source=rel["source"], target=rel["target"], label=rel["type"], arrows="to")
        for rel in viz_rels
    ]
    config = Config(
        width="100%",
        height=420,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#7C3AED",
    )
    _sagraph._agraph(
        data=json.dumps({"nodes": [node.to_dict() for node in nodes], "edges": [edge.to_dict() for edge in edges]}),
        config=json.dumps(config.__dict__),
        key=key,
    )


def format_sources(sources: list) -> str:
    lines = []
    for source in sources:
        if not isinstance(source, dict):
            lines.append(f"- `{source}`")
            continue
        title = source.get("title") or source.get("source") or "Unknown source"
        details = [f"**{title}**"]
        if source.get("collection"):
            details.append(f"collection: `{source['collection']}`")
        if source.get("source_type"):
            details.append(f"type: `{source['source_type']}`")
        lines.append("- " + " · ".join(details))
    return "\n".join(lines)


def compute_winner(graph_result: dict, vector_result: dict) -> str:
    graph_ok = bool(graph_result.get("graph_rows"))
    vector_ok = bool(vector_result.get("sources"))
    if graph_ok and not vector_ok:
        return "graph"
    if vector_ok and not graph_ok:
        return "vector"
    return "tie"


def log_feedback(question: str, pipeline: str, rating: str) -> None:
    path = Path("data/feedback.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "time": datetime.now().isoformat(),
                    "question": question,
                    "pipeline": pipeline,
                    "rating": rating,
                }
            )
            + "\n"
        )


def run_hybrid(question: str) -> None:
    st.session_state.hybrid_history.append({"role": "user", "content": question})
    with st.spinner("Routing and retrieving..."):
        result = answer_hybrid_langgraph(question)
    st.session_state.hybrid_history.append({"role": "assistant", **result})


def run_comparison(question: str) -> None:
    st.session_state.graph_history.append({"role": "user", "content": question})
    st.session_state.vector_history.append({"role": "user", "content": question})
    with st.spinner("Running graph and vector retrieval..."):
        graph_result = answer_question(question)
        vector_result = answer_question_vector(question)
    winner = compute_winner(graph_result, vector_result)
    st.session_state.graph_history.append(
        {"role": "assistant", "winner": winner, **graph_result}
    )
    st.session_state.vector_history.append(
        {"role": "assistant", "winner": winner, **vector_result}
    )


with st.sidebar:
    st.markdown("## 🧠 GraphMind")
    st.caption("Connected knowledge through GraphRAG and semantic search")
    if st.button("✨ New chat", use_container_width=True):
        for key in ("hybrid_history", "graph_history", "vector_history"):
            st.session_state[key] = []
        st.rerun()

    st.divider()
    st.markdown("**Graph inventory**")
    try:
        seed = json.loads(Path("data/processed/graph_seed.json").read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for node in seed.get("nodes", []):
            label = node.get("label", "Unknown")
            counts[label] = counts.get(label, 0) + 1
        stats = {
            "Documents": counts.get("Document", 0),
            "Chunks": counts.get("Chunk", 0),
            "Concepts": counts.get("Concept", 0),
            "Tools": counts.get("Tool", 0),
            "People": counts.get("Person", 0),
            "Organizations": counts.get("Organization", 0),
            "Graph nodes": len(seed.get("nodes", [])),
            "Relationships": len(seed.get("relationships", [])),
        }
    except (OSError, json.JSONDecodeError):
        stats = {"Documents": 0, "Chunks": 0, "Graph nodes": 0, "Relationships": 0}
    for label, value in stats.items():
        left, right = st.columns([3, 1])
        left.caption(label)
        right.markdown(f"**{value:,}**")

    st.divider()
    st.caption("Neo4j GraphRAG · Chroma semantic retrieval · LangGraph routing")


st.markdown("## 🧠 GraphMind")
st.caption("Explore relationships and source-grounded answers across your own documents.")

hybrid_col, compare_col, _ = st.columns([2, 2, 6])
if hybrid_col.button("Hybrid Assistant", use_container_width=True):
    st.session_state.mode = "hybrid"
if compare_col.button("Compare Pipelines", use_container_width=True):
    st.session_state.mode = "compare"

st.caption(f"LangGraph fallback: {'available' if LANGGRAPH_AVAILABLE else 'unavailable'}")
st.divider()

if not any((st.session_state.hybrid_history, st.session_state.graph_history, st.session_state.vector_history)):
    st.markdown("### Try a question")
    columns = st.columns(2)
    for index, demo in enumerate(DEMO_QUESTIONS):
        if columns[index % 2].button(demo, key=f"demo_{index}", use_container_width=True):
            if st.session_state.mode == "hybrid":
                run_hybrid(demo)
            else:
                run_comparison(demo)
            st.rerun()

if st.session_state.mode == "hybrid":
    for index, message in enumerate(st.session_state.hybrid_history):
        with st.chat_message(message["role"]):
            st.markdown(message.get("content") or message.get("answer", ""))
            if message["role"] == "assistant":
                st.caption(
                    f"Route: {message.get('route_label', 'Unknown')}"
                    + (" · fallback used" if message.get("fallback_used") else "")
                )
                if message.get("sources"):
                    with st.expander("Sources"):
                        st.markdown(format_sources(message["sources"]))
                if message.get("cypher"):
                    with st.expander("Cypher"):
                        st.code(message["cypher"], language="cypher")
                if message.get("viz_nodes"):
                    render_graph(message["viz_nodes"], message.get("viz_rels", []), f"hybrid_{index}")
                if message.get("related_content"):
                    with st.expander("Related source chunks"):
                        for item in message["related_content"]:
                            st.markdown(f"- **{item.get('name', 'Chunk')}** — `{item.get('source', '')}`")
    question = st.chat_input("Ask anything about your knowledge base...")
    if question:
        run_hybrid(question)
        st.rerun()
else:
    graph_col, vector_col = st.columns(2)
    with graph_col:
        st.markdown("### GraphRAG")
        for index, message in enumerate(st.session_state.graph_history):
            with st.chat_message(message["role"]):
                st.markdown(message.get("content") or message.get("answer", ""))
                if message["role"] == "assistant" and message.get("cypher"):
                    with st.expander("Cypher"):
                        st.code(message["cypher"], language="cypher")
                    if message.get("viz_nodes"):
                        render_graph(message["viz_nodes"], message.get("viz_rels", []), f"graph_{index}")
    with vector_col:
        st.markdown("### Vector RAG")
        for message in st.session_state.vector_history:
            with st.chat_message(message["role"]):
                st.markdown(message.get("content") or message.get("answer", ""))
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("Sources"):
                        st.markdown(format_sources(message["sources"]))
    question = st.chat_input("Ask both retrieval pipelines...")
    if question:
        run_comparison(question)
        st.rerun()

st.divider()
st.caption("GraphMind · Generic document knowledge graph · Neo4j + Chroma + LangGraph")
