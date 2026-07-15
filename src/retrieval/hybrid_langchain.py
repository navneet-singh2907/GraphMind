"""
Hybrid LangChain retriever — routes each question to GraphRAG or Vector RAG,
with an optional LangGraph fallback loop for low-confidence answers.
"""
import logging

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.retrieval.graph_rag import answer_question as _graph_answer
from src.retrieval.vector_rag import answer_question_vector as _vector_answer
from src.utils.config import NEBIUS_API_KEY, NEBIUS_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

# ── LangChain tools wrapping existing retrievers ──────────────────────────────

@tool
def graph_rag_tool(question: str) -> dict:
    """
    Query the Neo4j knowledge graph via Cypher.
    Best for: relationships, entities, concepts, tools, resources, and lists of
    connected items.
    """
    return _graph_answer(question)


@tool
def vector_rag_tool(question: str) -> dict:
    """
    Search document chunks via semantic similarity (ChromaDB + Nebius embeddings).
    Best for: explanations, summaries, what someone said, recommendations,
    quotes, source-backed details, and document evidence.
    """
    return _vector_answer(question)


# ── Router ────────────────────────────────────────────────────────────────────

_ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are a routing assistant for GraphMind, a connected knowledge system.

Given a user question, choose exactly ONE route:

GRAPH_RAG — use when the question asks about:
- Relationships, connections, lists of entities
- Documents, concepts, tools, projects, and resources
- Tools or concepts connected in the knowledge graph
- Which source or collection something belongs to
- Counts or structured lookups (e.g. "how many", "which ones")

VECTOR_RAG — use when the question asks about:
- Explanations or definitions ("what is", "how does")
- Summaries of what was covered ("what did we learn")
- What a specific source says or recommends
- Details, quotes, or evidence from documents
- Comparisons or recommendations

Examples:
"Which concepts are connected to GraphRAG?" -> GRAPH_RAG
"What tools does GraphMind use?" -> GRAPH_RAG
"Which resources discuss knowledge graphs?" -> GRAPH_RAG
"Summarize the retrieval architecture." -> VECTOR_RAG
"What evidence supports this recommendation?" -> VECTOR_RAG
"How does chunking affect RAG performance?" -> VECTOR_RAG

Respond with ONLY the route label and a one-line reason, in this exact format:
ROUTE: <GRAPH_RAG or VECTOR_RAG>
REASON: <one sentence explaining why>

Question: {question}"""
)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=NEBIUS_BASE_URL,
        temperature=0,
    )


def _heuristic_route(question: str) -> tuple[str, str] | None:
    """Fast fallback router for obvious demo intents."""
    lowered = question.lower()

    graph_markers = [
        "which documents",
        "which concepts",
        "which resources",
        "works at",
        "what tools",
        "which tools",
        "depends on",
        "references",
        "connected to",
        "relationship",
        "in the graph",
    ]
    if any(marker in lowered for marker in graph_markers):
        return "graph", "The question asks for structured relationships or entity lookups, so GraphRAG is the better route."

    vector_markers = [
        "what did",
        "what we learned",
        "what did we learn",
        "summarize",
        "explain",
        "what is",
        "how does",
        "how to",
        "why",
        "best",
        "say about",
        "told about",
        "discussed about",
        "source says",
        "document says",
    ]
    if any(marker in lowered for marker in vector_markers):
        return "vector", "The question needs explanation, summary, or source wording, so Vector RAG is the better route."

    return None


def route_question(question: str) -> tuple[str, str]:
    """Returns (route, reasoning) where route is 'graph' or 'vector'."""
    heuristic = _heuristic_route(question)
    if heuristic:
        return heuristic

    try:
        chain = _ROUTER_PROMPT | _get_llm() | StrOutputParser()
        raw = chain.invoke({"question": question}).strip()
    except Exception as exc:
        logger.warning("Hybrid router LLM failed; falling back to Vector RAG: %s", exc)
        return "vector", "The router LLM was unavailable, so the assistant used Vector RAG as the safest fallback for source-grounded answering."

    route = "vector"
    reasoning = ""
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("ROUTE:"):
            val = line.split(":", 1)[1].strip().upper()
            route = "graph" if "GRAPH" in val else "vector"
        elif line.startswith("REASON:"):
            reasoning = line.split(":", 1)[1].strip()

    return route, reasoning


# ── Confidence check ──────────────────────────────────────────────────────────

def _is_weak(result: dict, route: str) -> bool:
    answer = result.get("answer", "")
    low_confidence_phrases = [
        "couldn't find", "could not find", "no results", "not available",
        "i don't know", "unable to answer", "no information",
    ]
    if any(p in answer.lower() for p in low_confidence_phrases):
        return True
    if route == "graph" and not result.get("graph_rows"):
        return True
    if route == "vector" and not result.get("sources"):
        return True
    if len(answer.strip()) < 40:
        return True
    return False


# ── Phase 4: Hybrid answer (LangChain) ───────────────────────────────────────

def answer_hybrid(question: str) -> dict:
    """Route question to GraphRAG or Vector RAG, return unified answer dict."""
    route, reasoning = route_question(question)

    if route == "graph":
        result = _graph_answer(question)
        result["route"] = "graph"
        result["route_reasoning"] = reasoning
        result["route_label"] = "GraphRAG"
    else:
        result = _vector_answer(question)
        result["route"] = "vector"
        result["route_reasoning"] = reasoning
        result["route_label"] = "Vector RAG"

    return result


# ── Phase 6: LangGraph upgrade — confidence fallback ─────────────────────────

try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Optional

    class HybridState(TypedDict):
        question: str
        selected_route: str
        reasoning: str
        primary_result: Optional[dict]
        final_result: Optional[dict]
        fallback_used: bool

    def _node_route(state: HybridState) -> HybridState:
        r, reasoning = route_question(state["question"])
        return {**state, "selected_route": r, "reasoning": reasoning}

    def _node_primary(state: HybridState) -> HybridState:
        if state["selected_route"] == "graph":
            result = _graph_answer(state["question"])
        else:
            result = _vector_answer(state["question"])
        return {**state, "primary_result": result}

    def _node_check(state: HybridState) -> str:
        if _is_weak(state["primary_result"], state["selected_route"]):
            return "fallback"
        return "done"

    def _node_fallback(state: HybridState) -> HybridState:
        fallback_route = "vector" if state["selected_route"] == "graph" else "graph"
        if fallback_route == "graph":
            result = _graph_answer(state["question"])
        else:
            result = _vector_answer(state["question"])
        fallback_label = "GraphRAG" if fallback_route == "graph" else "Vector RAG"
        reasoning = f"The first route returned a weak answer, so LangGraph fell back to {fallback_label}."
        return {
            **state,
            "primary_result": result,
            "selected_route": fallback_route,
            "reasoning": reasoning,
            "fallback_used": True,
        }

    def _node_done(state: HybridState) -> HybridState:
        result = state["primary_result"]
        result["route"] = state["selected_route"]
        result["route_label"] = "GraphRAG" if state["selected_route"] == "graph" else "Vector RAG"
        result["route_reasoning"] = state["reasoning"]
        result["fallback_used"] = state.get("fallback_used", False)
        return {**state, "final_result": result}

    _graph = StateGraph(HybridState)
    _graph.add_node("route", _node_route)
    _graph.add_node("primary", _node_primary)
    _graph.add_node("fallback", _node_fallback)
    _graph.add_node("done", _node_done)
    _graph.set_entry_point("route")
    _graph.add_edge("route", "primary")
    _graph.add_conditional_edges("primary", _node_check, {"fallback": "fallback", "done": "done"})
    _graph.add_edge("fallback", "done")
    _graph.add_edge("done", END)
    _langgraph_app = _graph.compile()

    def answer_hybrid_langgraph(question: str) -> dict:
        """LangGraph version — routes, checks confidence, falls back if weak."""
        initial: HybridState = {
            "question": question,
            "selected_route": "",
            "reasoning": "",
            "primary_result": None,
            "final_result": None,
            "fallback_used": False,
        }
        final_state = _langgraph_app.invoke(initial)
        return final_state["final_result"]

    LANGGRAPH_AVAILABLE = True

except Exception as _lg_err:
    logger.warning("LangGraph not available: %s", _lg_err)
    LANGGRAPH_AVAILABLE = False

    def answer_hybrid_langgraph(question: str) -> dict:
        return answer_hybrid(question)
