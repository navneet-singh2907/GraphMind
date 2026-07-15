"""Build a generic GraphMind knowledge graph from processed document chunks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.neo4j_client import clear_graph, write_knowledge_graph
from src.graph.ontology import DEFAULT_ONTOLOGY
from src.utils.config import LLM_MODEL, NEBIUS_API_KEY, NEBIUS_BASE_URL


PROCESSED_DIR = Path("data/processed")
DOCUMENT_CHUNKS_PATH = PROCESSED_DIR / "document_chunks.jsonl"
GRAPH_SEED_PATH = PROCESSED_DIR / "graph_seed.json"

NODE_LABELS = list(DEFAULT_ONTOLOGY.node_labels)
EXTRACTABLE_NODE_LABELS = list(DEFAULT_ONTOLOGY.extraction_labels)
RELATIONSHIP_TYPES = list(DEFAULT_ONTOLOGY.relationship_types)

PROMPT = """Extract a knowledge graph from the document chunk below.

Return ONLY valid JSON in this exact shape:
{{
  "nodes": [
    {{"id": "stable_snake_case_id", "label": "Concept", "name": "Semantic search"}}
  ],
  "relationships": [
    {{"source": "stable_snake_case_id", "target": "other_id", "type": "RELATED_TO"}}
  ]
}}

Allowed node labels: {node_labels}
Allowed relationship types: {relationship_types}

Rules:
- Extract meaningful named people, organizations, topics, concepts, tools, projects, and resources.
- Do not create Document or Chunk nodes; GraphMind creates those deterministically.
- Do not extract generic filler words.
- Use canonical names and stable snake_case ids.
- Every relationship endpoint must refer to an id in the returned nodes.
- Return empty arrays when no useful graph material exists.

Document title: {title}
Collection: {collection}
Source type: {source_type}
Chunk id: {record_id}

Text:
{text}
"""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_graph_seed(path: Path = GRAPH_SEED_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_id(prefix: str, value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return f"{prefix}_{token}" if token else prefix


def base_nodes_and_relationships(records: list[dict]) -> tuple[dict, list[dict]]:
    nodes: dict[str, dict] = {}
    relationships: list[dict] = []

    for record in records:
        document_id = f"document_{record['document_id']}"
        chunk_id = record["id"]
        nodes[document_id] = {
            "id": document_id,
            "label": "Document",
            "name": record.get("title") or record.get("source_uri"),
            "source_uri": record.get("source_uri"),
            "source_type": record.get("source_type"),
            "collection": record.get("collection"),
        }
        nodes[chunk_id] = {
            "id": chunk_id,
            "label": "Chunk",
            "name": f"{record.get('title', 'Document')} chunk {record.get('chunk_index', 0) + 1}",
            "source_uri": record.get("source_uri"),
            "chunk_index": record.get("chunk_index"),
        }
        relationships.append({"source": chunk_id, "target": document_id, "type": "PART_OF"})

    return nodes, relationships


def dedupe_relationships(relationships: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for relationship in relationships:
        key = (
            relationship.get("source"), relationship.get("target"),
            relationship.get("type"),
        )
        if key not in seen:
            seen.add(key)
            result.append(relationship)
    return result


def extract_record(client: OpenAI, record: dict) -> dict:
    prompt = PROMPT.format(
        node_labels=", ".join(EXTRACTABLE_NODE_LABELS),
        relationship_types=", ".join(RELATIONSHIP_TYPES),
        title=record.get("title"),
        collection=record.get("collection"),
        source_type=record.get("source_type"),
        record_id=record.get("id"),
        text=record.get("content", "")[:6000],
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def build_graph_seed(
    limit: int | None = None,
    skip: int = 0,
    base_only: bool = False,
    extract_source_type: str | None = None,
    extract_collection: str | None = None,
    merge_existing: bool = False,
    checkpoint_every: int = 25,
) -> dict:
    records = read_jsonl(DOCUMENT_CHUNKS_PATH)
    nodes, relationships = base_nodes_and_relationships(records)

    if merge_existing and GRAPH_SEED_PATH.exists():
        existing = read_graph_seed()
        for node in existing.get("nodes", []):
            if node.get("label") in NODE_LABELS:
                nodes[node["id"]] = node
        relationships.extend(
            rel for rel in existing.get("relationships", [])
            if rel.get("type") in RELATIONSHIP_TYPES
        )

    extraction_records = records
    if extract_source_type:
        extraction_records = [
            row for row in extraction_records if row.get("source_type") == extract_source_type
        ]
    if extract_collection:
        extraction_records = [
            row for row in extraction_records if row.get("collection") == extract_collection
        ]
    extraction_records = extraction_records[skip:]
    if limit is not None:
        extraction_records = extraction_records[:limit]

    if not base_only:
        client = OpenAI(api_key=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)
        for index, record in enumerate(extraction_records, start=1):
            print(f"Extracting graph items {index}/{len(extraction_records)}: {record['id']}")
            try:
                extracted = extract_record(client, record)
            except Exception as exc:
                print(f"  skipped {record['id']}: {exc}")
                continue

            local_ids = set()
            for node in extracted.get("nodes", []):
                label, name, node_id = node.get("label"), node.get("name"), node.get("id")
                if label not in EXTRACTABLE_NODE_LABELS or not name or not node_id:
                    continue
                nodes[node_id] = {"id": node_id, "label": label, "name": name}
                local_ids.add(node_id)

            for relationship in extracted.get("relationships", []):
                if (
                    relationship.get("type") in RELATIONSHIP_TYPES
                    and relationship.get("source") in local_ids
                    and relationship.get("target") in local_ids
                ):
                    relationships.append(relationship)

            for node_id in local_ids:
                relationships.append(
                    {"source": record["id"], "target": node_id, "type": "DISCUSSES"}
                )

            if checkpoint_every and index % checkpoint_every == 0:
                graph = {
                    "nodes": list(nodes.values()),
                    "relationships": dedupe_relationships(relationships),
                }
                GRAPH_SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
                GRAPH_SEED_PATH.write_text(
                    json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
                )

    graph = {
        "nodes": list(nodes.values()),
        "relationships": dedupe_relationships(relationships),
    }
    GRAPH_SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_SEED_PATH.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GraphMind knowledge graph.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--load-existing", action="store_true")
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument("--extract-source-type")
    parser.add_argument("--extract-collection")
    parser.add_argument("--write-neo4j", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    graph = read_graph_seed() if args.load_existing else build_graph_seed(
        limit=args.limit,
        skip=args.skip,
        base_only=args.base_only,
        extract_source_type=args.extract_source_type,
        extract_collection=args.extract_collection,
        merge_existing=args.merge_existing,
        checkpoint_every=args.checkpoint_every,
    )
    print(
        f"Wrote {len(graph['nodes'])} nodes and {len(graph['relationships'])} "
        f"relationships to {GRAPH_SEED_PATH}"
    )

    if args.write_neo4j:
        if args.clear:
            clear_graph()
        write_knowledge_graph(graph["nodes"], graph["relationships"])


if __name__ == "__main__":
    main()
