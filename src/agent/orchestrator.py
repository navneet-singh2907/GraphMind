"""Bounded plan → retrieve → verify → retry agent for GraphMind."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.retrieval.models import Evidence
from src.retrieval.tools import TOOL_DESCRIPTIONS, execute_tool
from src.utils.config import AGENT_MAX_ATTEMPTS, LLM_MODEL, NEBIUS_API_KEY, NEBIUS_BASE_URL


@dataclass(slots=True)
class ToolCall:
    tool: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class RetrievalPlan:
    reasoning: str
    calls: list[ToolCall]


@dataclass(slots=True)
class Verification:
    sufficient: bool
    confidence: float
    missing: str = ""


@dataclass(slots=True)
class AgentResult:
    question: str
    answer: str
    evidence: list[Evidence]
    plan: RetrievalPlan
    verification: Verification
    attempts: int
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        sources = []
        seen = set()
        for item in self.evidence:
            key = (item.source_uri, item.title, item.tool)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "source": item.source_uri or item.tool,
                    "title": item.title or item.tool,
                    "retriever": item.tool,
                    **({"score": item.score} if item.score is not None else {}),
                }
            )
        return {
            "question": self.question,
            "answer": self.answer,
            "route": "agentic",
            "route_label": "Agentic RAG",
            "fallback_used": self.attempts > 1,
            "sources": sources,
            "evidence_count": len(self.evidence),
            "plan": {
                "reasoning": self.plan.reasoning,
                "calls": [asdict(call) for call in self.plan.calls],
            },
            "verification": asdict(self.verification),
            "attempts": self.attempts,
            "trace": self.trace,
        }


def _client():
    if not NEBIUS_API_KEY:
        raise RuntimeError("NEBIUS_API_KEY is not configured")
    from openai import OpenAI

    return OpenAI(api_key=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)


def _json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lower().startswith("json"):
            text = text[4:]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model response did not contain a JSON object")
    return json.loads(match.group(0))


def _fallback_plan(question: str, collection: str | None = None) -> RetrievalPlan:
    lowered = question.lower()
    calls = []
    common = {"query": question, "k": 8}
    if collection:
        common["collection"] = collection
    if any(marker in lowered for marker in ["connected", "relationship", "depends", "uses", "which"]):
        calls.append(ToolCall("graph_search", {"query": question, "k": 20}))
    calls.append(ToolCall("vector_search", dict(common)))
    calls.append(ToolCall("keyword_search", dict(common)))
    return RetrievalPlan(
        reasoning="Fallback plan combines semantic and lexical evidence, plus graph traversal for relational intent.",
        calls=calls,
    )


def plan_question(question: str, collection: str | None = None) -> RetrievalPlan:
    tool_text = "\n".join(f"- {name}: {description}" for name, description in TOOL_DESCRIPTIONS.items())
    prompt = f"""You are the retrieval planner for GraphMind.
Create a small, cost-aware plan for answering the question using the available tools.

Tools:
{tool_text}

Rules:
- Use 1 to 3 calls.
- Combine vector_search and keyword_search when wording may vary or exact terms matter.
- Use graph_search for relationship or multi-hop questions.
- Use metadata_search for collection, type, title, or inventory questions.
- Use inspect_source only when an exact source_uri is already known.
- Tool arguments must include query for search tools and may include k, collection, or source_type.

Return only JSON:
{{"reasoning":"...","calls":[{{"tool":"vector_search","arguments":{{"query":"...","k":8}}}}]}}

Question: {question}
Collection constraint: {collection or "none"}
"""
    try:
        response = _client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        data = _json_object(response.choices[0].message.content)
        calls = []
        for item in data.get("calls", [])[:3]:
            tool = item.get("tool")
            if tool not in TOOL_DESCRIPTIONS:
                continue
            arguments = dict(item.get("arguments") or {})
            if tool != "inspect_source":
                arguments.setdefault("query", question)
            arguments.setdefault("k", 8)
            if collection and tool in {"vector_search", "keyword_search", "metadata_search"}:
                arguments["collection"] = collection
            calls.append(ToolCall(tool=tool, arguments=arguments))
        if not calls:
            raise ValueError("Planner returned no valid tool calls")
        return RetrievalPlan(reasoning=data.get("reasoning", "LLM retrieval plan"), calls=calls)
    except Exception:
        return _fallback_plan(question, collection)


def _dedupe(evidence: list[Evidence], limit: int = 20) -> list[Evidence]:
    result = []
    seen = set()
    for item in evidence:
        key = (item.source_uri, item.content[:300])
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def execute_plan(plan: RetrievalPlan, trace: list[dict]) -> list[Evidence]:
    evidence = []
    for call in plan.calls:
        try:
            results = execute_tool(call.tool, call.arguments)
            evidence.extend(results)
            trace.append(
                {"stage": "retrieve", "tool": call.tool, "arguments": call.arguments, "results": len(results)}
            )
        except Exception as exc:
            trace.append(
                {"stage": "retrieve", "tool": call.tool, "arguments": call.arguments, "error": str(exc)}
            )
    return _dedupe(evidence)


def _evidence_context(evidence: list[Evidence], limit: int = 12) -> str:
    blocks = []
    for index, item in enumerate(evidence[:limit], start=1):
        label = item.title or item.source_uri or item.tool
        blocks.append(f"[S{index}] Tool={item.tool}; Source={label}\n{item.content[:1800]}")
    return "\n\n".join(blocks)


def verify_evidence(question: str, evidence: list[Evidence]) -> Verification:
    if not evidence:
        return Verification(False, 0.0, "No evidence was retrieved.")
    prompt = f"""Evaluate whether the evidence is sufficient to answer the question accurately.
Return only JSON: {{"sufficient":true,"confidence":0.0,"missing":""}}
Confidence must be between 0 and 1. Require evidence that directly supports the answer.

Question: {question}

Evidence:
{_evidence_context(evidence)}
"""
    try:
        response = _client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        data = _json_object(response.choices[0].message.content)
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0))))
        return Verification(bool(data.get("sufficient")) and confidence >= 0.55, confidence, data.get("missing", ""))
    except Exception:
        source_count = len({item.source_uri for item in evidence if item.source_uri})
        confidence = min(0.85, 0.35 + 0.1 * len(evidence) + 0.1 * source_count)
        return Verification(len(evidence) >= 2, confidence, "")


def _retry_plan(
    question: str,
    previous: RetrievalPlan,
    verification: Verification,
    collection: str | None,
) -> RetrievalPlan:
    used = {call.tool for call in previous.calls}
    calls = []
    query = f"{question} {verification.missing}".strip()
    for candidate in ("keyword_search", "vector_search", "graph_search", "metadata_search"):
        if candidate not in used:
            arguments: dict[str, Any] = {"query": query, "k": 12}
            if collection and candidate in {"keyword_search", "vector_search", "metadata_search"}:
                arguments["collection"] = collection
            calls.append(ToolCall(candidate, arguments))
            break
    if not calls:
        calls.append(ToolCall("keyword_search", {"query": query, "k": 16, **({"collection": collection} if collection else {})}))
    return RetrievalPlan(
        reasoning=f"Verification found insufficient evidence. Expanding retrieval for: {verification.missing or 'the original question'}",
        calls=calls,
    )


def synthesize_answer(question: str, evidence: list[Evidence], verification: Verification) -> str:
    if not evidence:
        return "I could not find evidence for this question in the indexed sources."
    prompt = f"""Answer the question using only the evidence below.
Use inline citations like [S1] and [S2]. Do not cite unsupported claims.
If evidence conflicts, describe the conflict. If it remains insufficient, state the limitation.

Question: {question}
Evidence confidence: {verification.confidence:.2f}

{_evidence_context(evidence)}
"""
    try:
        response = _client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        previews = "\n\n".join(
            f"[S{index}] {item.content[:500]}" for index, item in enumerate(evidence[:3], start=1)
        )
        return f"Model synthesis was unavailable. The strongest retrieved evidence was:\n\n{previews}"


def answer_agentic(
    question: str,
    collection: str | None = None,
    max_attempts: int = AGENT_MAX_ATTEMPTS,
) -> dict:
    trace: list[dict] = []
    plan = plan_question(question, collection)
    trace.append({"stage": "plan", "reasoning": plan.reasoning, "calls": [asdict(call) for call in plan.calls]})
    evidence: list[Evidence] = []
    verification = Verification(False, 0.0, "Not verified")

    for attempt in range(1, max(1, max_attempts) + 1):
        evidence = _dedupe(evidence + execute_plan(plan, trace))
        verification = verify_evidence(question, evidence)
        trace.append({"stage": "verify", "attempt": attempt, **asdict(verification)})
        if verification.sufficient or attempt >= max_attempts:
            break
        plan = _retry_plan(question, plan, verification, collection)
        trace.append({"stage": "replan", "reasoning": plan.reasoning, "calls": [asdict(call) for call in plan.calls]})

    answer = synthesize_answer(question, evidence, verification)
    return AgentResult(
        question=question,
        answer=answer,
        evidence=evidence,
        plan=plan,
        verification=verification,
        attempts=attempt,
        trace=trace,
    ).to_dict()
