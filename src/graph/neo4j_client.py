"""Neo4j access helpers for the generic GraphMind schema."""

from neo4j import GraphDatabase

from src.graph.ontology import DEFAULT_ONTOLOGY
from src.utils.config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME


ALLOWED_NODE_LABELS = frozenset(DEFAULT_ONTOLOGY.node_labels)
ALLOWED_REL_TYPES = frozenset(DEFAULT_ONTOLOGY.relationship_types)


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def get_graph_inventory() -> dict:
    """Return total and per-label counts for the configured Neo4j database."""
    driver = get_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            summary = session.run(
                """
                MATCH (node)
                WITH count(node) AS nodes
                OPTIONAL MATCH ()-[relationship]->()
                RETURN nodes, count(relationship) AS relationships
                """
            ).single()
            label_rows = session.run(
                """
                MATCH (node)
                UNWIND labels(node) AS label
                RETURN label, count(node) AS count
                ORDER BY label
                """
            )
            return {
                "nodes": int(summary["nodes"]) if summary else 0,
                "relationships": int(summary["relationships"]) if summary else 0,
                "labels": {row["label"]: int(row["count"]) for row in label_rows},
            }
    finally:
        driver.close()


def get_related_content(entity_names: list[str], limit: int = 5) -> list[dict]:
    """Return source chunks that discuss any of the supplied entities."""
    names = list({name for name in entity_names if isinstance(name, str) and name.strip()})[:10]
    if not names:
        return []

    driver = get_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (chunk:Chunk)-[:DISCUSSES]->(entity)
                WHERE entity.name IN $names
                RETURN DISTINCT chunk.name AS name, chunk.source_uri AS source
                LIMIT $limit
                """,
                names=names,
                limit=limit,
            )
            return [dict(record) for record in result]
    finally:
        driver.close()


def get_subgraph(entity_names: list[str]) -> tuple[list[dict], list[dict]]:
    names = list({name for name in entity_names if isinstance(name, str) and name.strip()})[:10]
    if not names:
        return [], []

    driver = get_driver()
    nodes: dict[str, dict] = {}
    relationships: list[dict] = []
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (a)-[r]-(b)
                WHERE a.name IN $names OR b.name IN $names
                RETURN a, r, b LIMIT 80
                """,
                names=names,
            )
            for record in result:
                for key in ("a", "b"):
                    node = record[key]
                    element_id = str(node.element_id)
                    if element_id not in nodes:
                        label = list(node.labels)[0] if node.labels else "Node"
                        nodes[element_id] = {
                            "id": element_id,
                            "label": label,
                            "name": node.get("name", element_id),
                        }
                relationship = record["r"]
                relationships.append(
                    {
                        "source": str(relationship.start_node.element_id),
                        "target": str(relationship.end_node.element_id),
                        "type": relationship.type,
                    }
                )
    finally:
        driver.close()
    return list(nodes.values()), relationships


def _safe_label(label: str) -> str:
    return DEFAULT_ONTOLOGY.validate_node_label(label)


def _safe_rel_type(rel_type: str) -> str:
    return DEFAULT_ONTOLOGY.validate_relationship_type(rel_type)


def write_graph(entities: list, relationships: list) -> None:
    """Write the small name-based demo graph used by ``seed.py``."""
    driver = get_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for entity in entities:
                label = _safe_label(entity["type"])
                session.run(f"MERGE (n:{label} {{name: $name}})", name=entity["name"])
            for relationship in relationships:
                rel_type = _safe_rel_type(relationship["type"])
                session.run(
                    f"""
                    MATCH (a {{name: $source}})
                    MATCH (b {{name: $target}})
                    MERGE (a)-[:{rel_type}]->(b)
                    """,
                    source=relationship["source"],
                    target=relationship["target"],
                )
    finally:
        driver.close()
    print(f"  Written {len(entities)} entities and {len(relationships)} relationships to Neo4j.")


def write_knowledge_graph(nodes: list, relationships: list) -> None:
    driver = get_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            nodes_by_label: dict[str, list[dict]] = {}
            id_to_label: dict[str, str] = {}
            for node in nodes:
                label = _safe_label(node["label"])
                properties = {key: value for key, value in node.items() if key != "label" and value is not None}
                nodes_by_label.setdefault(label, []).append({"id": node["id"], "props": properties})
                id_to_label[node["id"]] = label

            for label, rows in nodes_by_label.items():
                session.run(f"CREATE INDEX {label}_id IF NOT EXISTS FOR (n:{label}) ON (n.id)")
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MERGE (n:{label} {{id: row.id}})
                    SET n += row.props
                    """,
                    rows=rows,
                )

            grouped: dict[tuple[str, str, str], list[dict]] = {}
            for relationship in relationships:
                rel_type = _safe_rel_type(relationship["type"])
                source_label = id_to_label.get(relationship["source"])
                target_label = id_to_label.get(relationship["target"])
                if source_label and target_label:
                    grouped.setdefault((rel_type, source_label, target_label), []).append(
                        {"source": relationship["source"], "target": relationship["target"]}
                    )

            for (rel_type, source_label, target_label), rows in grouped.items():
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (a:{source_label} {{id: row.source}})
                    MATCH (b:{target_label} {{id: row.target}})
                    MERGE (a)-[:{rel_type}]->(b)
                    """,
                    rows=rows,
                )
    finally:
        driver.close()
    print(f"  Written {len(nodes)} nodes and {len(relationships)} relationships to Neo4j.")


def clear_graph() -> None:
    driver = get_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()
    print("  Graph cleared.")
