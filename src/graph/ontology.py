"""Configurable ontology loading and validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONTOLOGY_PATH = ROOT / "config" / "default_ontology.json"


@dataclass(frozen=True, slots=True)
class Ontology:
    name: str
    node_labels: dict[str, str]
    extraction_labels: tuple[str, ...]
    relationship_types: dict[str, str]

    def validate_node_label(self, label: str) -> str:
        if label not in self.node_labels:
            raise ValueError(f"Unsupported node label: {label}")
        return label

    def validate_relationship_type(self, relationship_type: str) -> str:
        if relationship_type not in self.relationship_types:
            raise ValueError(f"Unsupported relationship type: {relationship_type}")
        return relationship_type

    def schema_text(self) -> str:
        node_lines = "\n".join(
            f"- {label}: {description}" for label, description in self.node_labels.items()
        )
        relationship_lines = "\n".join(
            f"- {name}: {description}" for name, description in self.relationship_types.items()
        )
        return f"Node labels:\n{node_lines}\n\nRelationship types:\n{relationship_lines}"


def load_ontology(path: str | Path | None = None) -> Ontology:
    configured = path or os.getenv("ONTOLOGY_PATH")
    ontology_path = Path(configured).expanduser() if configured else DEFAULT_ONTOLOGY_PATH
    if not ontology_path.is_absolute():
        ontology_path = ROOT / ontology_path
    data = json.loads(ontology_path.read_text(encoding="utf-8"))
    ontology = Ontology(
        name=data.get("name", ontology_path.stem),
        node_labels=dict(data["node_labels"]),
        extraction_labels=tuple(data["extraction_labels"]),
        relationship_types=dict(data["relationship_types"]),
    )
    missing = set(ontology.extraction_labels) - set(ontology.node_labels)
    if missing:
        raise ValueError(f"Extraction labels missing from node_labels: {sorted(missing)}")
    for required in ("Document", "Chunk"):
        if required not in ontology.node_labels:
            raise ValueError(f"Ontology must define the structural label {required}")
    for required in ("PART_OF", "DISCUSSES"):
        if required not in ontology.relationship_types:
            raise ValueError(f"Ontology must define the structural relationship {required}")
    return ontology


DEFAULT_ONTOLOGY = load_ontology()
