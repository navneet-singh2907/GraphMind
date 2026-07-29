"""Generic Neo4j GraphRAG retrieval for GraphMind."""

from __future__ import annotations

import re

from neo4j import GraphDatabase
from openai import OpenAI

from src.graph.neo4j_client import get_related_content, get_subgraph
from src.graph.ontology import DEFAULT_ONTOLOGY
from src.utils.config import (
    LLM_MODEL,
    NEBIUS_API_KEY,
    NEBIUS_BASE_URL,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    MODEL_MAX_RETRIES,
    MODEL_TIMEOUT_SECONDS,
)


def _client() -> OpenAI:
    if not NEBIUS_API_KEY:
        raise RuntimeError("NEBIUS_API_KEY is not configured")
    return OpenAI(
        api_key=NEBIUS_API_KEY,
        base_url=NEBIUS_BASE_URL,
        timeout=MODEL_TIMEOUT_SECONDS,
        max_retries=MODEL_MAX_RETRIES,
    )

GRAPH_SCHEMA = (
    DEFAULT_ONTOLOGY.schema_text()
    + "\n\nStructural properties:\n"
    + "- Document {id, name, source_uri, source_type, collection}\n"
    + "- Chunk {id, name, source_uri, chunk_index}\n"
    + "The graph contents depend entirely on the indexed documents."
)

ALLOWED_GRAPH_PROPERTIES = frozenset(
    {"id", "name", "source_uri", "source_type", "collection", "chunk_index"}
)

CYPHER_PROMPT = """You are a Neo4j Cypher expert. Create one read-only query that answers the question.

Graph schema:
{schema}

Examples:

Q: Which concepts are connected to GraphRAG?
MATCH (concept:Concept)-[relationship]-(related)
WHERE toLower(concept.name) CONTAINS "graphrag"
RETURN concept.name AS concept, type(relationship) AS relationship, related.name AS related LIMIT 20

Q: Which documents discuss vector search?
MATCH (document:Document)<-[:PART_OF]-(chunk:Chunk)-[:DISCUSSES]->(concept)
WHERE toLower(concept.name) CONTAINS "vector search"
RETURN DISTINCT document.name AS document, document.source_uri AS source LIMIT 20

Q: What tool does Example System use?
MATCH (entity)-[relationship:USES]->(tool:Tool)
WHERE toLower(entity.name) CONTAINS "example system"
RETURN entity.name AS entity, type(relationship) AS relationship, tool.name AS tool LIMIT 20

Rules:
- Return only raw Cypher, without markdown or explanation.
- Use only MATCH, OPTIONAL MATCH, WITH, WHERE, UNWIND, and RETURN clauses.
- Never create, merge, update, delete, load, or call procedures.
- Use the exact labels and relationship types from the schema.
- Limit results to at most 20 rows.
- Prefer case-insensitive CONTAINS when names may vary.
- When an entity is identified by name but its label is not explicitly stated in the question, always
  omit the label and match `(entity)` by its `name` property. Do not infer Project, Document, Tool,
  or another label from capitalization or wording.
- Knowledge entities can connect directly. Do not force a named entity through Document or Chunk
  unless the question explicitly asks about documents, sources, or passages.
- For questions asking what a named entity uses, first use the direct label-free pattern
  `MATCH (entity)-[relationship:USES]->(tool:Tool)` and filter `entity.name`
  case-insensitively. Do not route this pattern through Document or Chunk.
- For variable-length traversal, bind the path in MATCH (for example `MATCH path = (a)-[:USES*1..2]-(b)`) before calling `length(path)`; never call `length()` on a parenthesized pattern expression.
- Bind relationship variables explicitly (for example `(a)-[relationship:USES]->(b)`). The `type()` function accepts a relationship variable only; never pass a node or path variable to `type()`.
- Return `RETURN "NO_MATCH" AS result` only when the schema cannot answer the question.

Question: {question}
{correction}
"""

ANSWER_PROMPT = """You are GraphMind, a helpful assistant for a connected knowledge base.
Answer the question using only the returned graph data.

Cypher:
{cypher}

Graph data:
{graph_data}

Question: {question}

Name specific entities and sources when available. If the graph returned no useful data, say so clearly.
"""


def generate_cypher(question: str, correction: str = "") -> str:
    prompt = CYPHER_PROMPT.format(
        schema=GRAPH_SCHEMA,
        question=question,
        correction=correction,
    )
    response = _client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("cypher"):
            raw = raw[6:]
    return raw.strip()


def _validate_readonly_cypher(cypher: str) -> tuple[bool, str | None]:
    stripped = cypher.strip()
    if not stripped:
        return False, "The graph query generator returned an empty query."

    first_word = stripped.split(maxsplit=1)[0].upper()
    if first_word not in {"MATCH", "OPTIONAL", "WITH", "RETURN", "UNWIND"}:
        return False, "GraphMind only runs read-only Cypher queries."

    padded = f" {stripped.upper()} "
    blocked = [
        " CREATE ", " MERGE ", " DELETE ", " SET ", " REMOVE ", " DROP ",
        " LOAD CSV ", " DETACH ", " CALL ", " FOREACH ",
    ]
    if any(token in padded for token in blocked):
        return False, "GraphMind refused a graph mutation or procedure call."

    labels = set(
        re.findall(r"\([^)]*?:\s*([A-Za-z_][A-Za-z0-9_]*)", stripped)
    )
    unknown_labels = labels - set(DEFAULT_ONTOLOGY.node_labels)
    if unknown_labels:
        return False, f"Unsupported node labels: {sorted(unknown_labels)}"

    relationship_types = set(
        re.findall(r"\[[^\]]*?:\s*([A-Za-z_][A-Za-z0-9_]*)", stripped)
    )
    unknown_relationship_types = relationship_types - set(DEFAULT_ONTOLOGY.relationship_types)
    if unknown_relationship_types:
        return False, (
            "Unsupported relationship types: "
            f"{sorted(unknown_relationship_types)}"
        )

    properties = set(
        re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)", stripped)
    )
    unknown_properties = properties - ALLOWED_GRAPH_PROPERTIES
    if unknown_properties:
        return False, f"Unsupported graph properties: {sorted(unknown_properties)}"

    return True, None


def run_cypher(cypher: str) -> list[dict]:
    ok, reason = _validate_readonly_cypher(cypher)
    if not ok:
        return [{"error": reason}]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            return [dict(record) for record in session.run(cypher)]
    finally:
        driver.close()


def query_graph(question: str, max_attempts: int = 3) -> tuple[str, list[dict]]:
    """Generate and execute read-only Cypher, correcting one database-rejected query."""
    correction = ""
    last_error: Exception | None = None
    for attempt in range(max(1, max_attempts)):
        cypher = generate_cypher(question, correction=correction)
        try:
            ok, reason = _validate_readonly_cypher(cypher)
            if not ok:
                raise ValueError(reason)
            rows = run_cypher(cypher)
            if rows or attempt == max(1, max_attempts) - 1:
                return cypher, rows
            correction = (
                "The previous query was valid but returned no rows. Broaden it while keeping the original intent: "
                "remove uncertain node labels, allow relevant relationship types, and match named entities "
                "case-insensitively.\n"
                f"Previous query: {cypher[:2000]}"
            )
        except Exception as exc:
            last_error = exc
            correction = (
                "The previous query was rejected by Neo4j. Correct it using the schema and rules above.\n"
                f"Previous query: {cypher[:2000]}\n"
                f"Neo4j error: {str(exc)[:1200]}"
            )
    raise RuntimeError(f"Neo4j rejected the generated query after {max_attempts} attempts: {last_error}")


def answer_question(question: str) -> dict:
    try:
        cypher, rows = query_graph(question)
    except Exception as exc:
        message = f"Graph retrieval is unavailable: {exc}"
        return {
            "question": question,
            "cypher": "",
            "graph_rows": [],
            "answer": message,
            "viz_nodes": [],
            "viz_rels": [],
            "related_content": [],
        }
    ok, reason = _validate_readonly_cypher(cypher)
    if not ok:
        return {
            "question": question,
            "cypher": cypher,
            "graph_rows": [{"error": reason}],
            "answer": reason,
            "viz_nodes": [],
            "viz_rels": [],
            "related_content": [],
        }

    graph_data = str(rows) if rows else "No results found."
    response = _client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(
                    cypher=cypher, graph_data=graph_data, question=question
                ),
            }
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()

    entity_names = [str(value) for row in rows for value in row.values() if isinstance(value, str)]
    viz_nodes, viz_rels = get_subgraph(entity_names)
    return {
        "question": question,
        "cypher": cypher,
        "graph_rows": rows,
        "answer": answer,
        "viz_nodes": viz_nodes,
        "viz_rels": viz_rels,
        "related_content": get_related_content(entity_names),
    }
