"""Generic Neo4j GraphRAG retrieval for GraphMind."""

from __future__ import annotations

from neo4j import GraphDatabase
from openai import OpenAI

from src.graph.neo4j_client import get_related_content, get_subgraph
from src.utils.config import (
    LLM_MODEL,
    NEBIUS_API_KEY,
    NEBIUS_BASE_URL,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)


client = OpenAI(api_key=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)

GRAPH_SCHEMA = """
Node labels:
- Document {id, name, source_path, source_type, collection}
- Chunk {id, name, source_path, chunk_index}
- Person {id, name}
- Organization {id, name}
- Topic {id, name}
- Concept {id, name}
- Tool {id, name}
- Project {id, name}
- Resource {id, name}

Relationship types:
- (Chunk)-[:PART_OF]->(Document)
- (Chunk)-[:DISCUSSES]->(Person|Organization|Topic|Concept|Tool|Project|Resource)
- (Document|Chunk)-[:MENTIONS|REFERENCES]->(Person|Organization|Topic|Concept|Tool|Project|Resource)
- (Person)-[:WORKS_AT]->(Organization)
- (Project)-[:CREATED_BY]->(Person|Organization)
- (Project|Tool)-[:USES|DEPENDS_ON]->(Tool|Resource|Concept)
- (Project|Resource)-[:APPLIES]->(Concept|Topic|Tool)
- Any knowledge entity may use RELATED_TO, COMPARES_WITH, or REQUIRES where supported.

The graph contents depend entirely on the documents indexed by the user.
"""

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
RETURN DISTINCT document.name AS document, document.source_path AS source LIMIT 20

Q: What tools does GraphMind use?
MATCH (project:Project)-[:USES]->(tool:Tool)
WHERE toLower(project.name) CONTAINS "graphmind"
RETURN project.name AS project, tool.name AS tool LIMIT 20

Rules:
- Return only raw Cypher, without markdown or explanation.
- Use only MATCH, OPTIONAL MATCH, WITH, WHERE, UNWIND, and RETURN clauses.
- Never create, merge, update, delete, load, or call procedures.
- Use the exact labels and relationship types from the schema.
- Limit results to at most 20 rows.
- Prefer case-insensitive CONTAINS when names may vary.
- Return `RETURN "NO_MATCH" AS result` only when the schema cannot answer the question.

Question: {question}
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


def generate_cypher(question: str) -> str:
    prompt = CYPHER_PROMPT.format(schema=GRAPH_SCHEMA, question=question)
    response = client.chat.completions.create(
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


def answer_question(question: str) -> dict:
    cypher = generate_cypher(question)
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

    rows = run_cypher(cypher)
    graph_data = str(rows) if rows else "No results found."
    response = client.chat.completions.create(
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
