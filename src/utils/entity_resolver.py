from difflib import SequenceMatcher
from src.graph.seed import ENTITIES

SIMILARITY_THRESHOLD = 0.85


def build_entity_index() -> list[str]:
    return [e["name"] for e in ENTITIES]


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def resolve_name(name: str, index: list[str]) -> str:
    best_match = name
    best_score = 0.0

    for canonical in index:
        score = _similarity(name, canonical)
        if score > best_score:
            best_score = score
            best_match = canonical

    if best_score >= SIMILARITY_THRESHOLD:
        if best_match != name:
            print(f"    Resolved: '{name}' → '{best_match}' (score: {best_score:.2f})")
        return best_match

    return name


def resolve_entities(graph_elements: dict) -> dict:
    index = build_entity_index()

    resolved_entities = {}
    for entity in graph_elements["entities"]:
        resolved_name = resolve_name(entity["name"], index)
        key = (resolved_name, entity["type"])
        resolved_entities[key] = {"name": resolved_name, "type": entity["type"]}

    resolved_relationships = []
    for rel in graph_elements["relationships"]:
        resolved_relationships.append({
            "source": resolve_name(rel["source"], index),
            "target": resolve_name(rel["target"], index),
            "type": rel["type"],
        })

    return {
        "entities": list(resolved_entities.values()),
        "relationships": resolved_relationships,
    }
