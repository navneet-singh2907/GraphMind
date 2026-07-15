"""Small, generic demo graph for local GraphMind development."""

from src.graph.neo4j_client import clear_graph, write_graph


ENTITIES = [
    {"name": "GraphMind", "type": "Project"},
    {"name": "GraphRAG", "type": "Concept"},
    {"name": "Hybrid RAG", "type": "Concept"},
    {"name": "Embeddings", "type": "Concept"},
    {"name": "Chunking", "type": "Concept"},
    {"name": "Vector Search", "type": "Concept"},
    {"name": "Knowledge Graph", "type": "Concept"},
    {"name": "LangChain", "type": "Tool"},
    {"name": "LangGraph", "type": "Tool"},
    {"name": "Neo4j", "type": "Tool"},
    {"name": "Chroma", "type": "Tool"},
    {"name": "Python", "type": "Tool"},
    {"name": "Architecture Notes", "type": "Resource"},
    {"name": "Retrieval Guide", "type": "Resource"},
]


RELATIONSHIPS = [
    {"source": "GraphMind", "target": "LangChain", "type": "USES"},
    {"source": "GraphMind", "target": "LangGraph", "type": "USES"},
    {"source": "GraphMind", "target": "Neo4j", "type": "USES"},
    {"source": "GraphMind", "target": "Chroma", "type": "USES"},
    {"source": "GraphMind", "target": "Python", "type": "USES"},
    {"source": "GraphMind", "target": "GraphRAG", "type": "APPLIES"},
    {"source": "GraphMind", "target": "Hybrid RAG", "type": "APPLIES"},
    {"source": "LangGraph", "target": "LangChain", "type": "BUILT_ON"},
    {"source": "Neo4j", "target": "Knowledge Graph", "type": "RELATED_TO"},
    {"source": "Chroma", "target": "Vector Search", "type": "RELATED_TO"},
    {"source": "Embeddings", "target": "Vector Search", "type": "RELATED_TO"},
    {"source": "Chunking", "target": "Vector Search", "type": "RELATED_TO"},
    {"source": "Architecture Notes", "target": "Knowledge Graph", "type": "DISCUSSES"},
    {"source": "Retrieval Guide", "target": "Hybrid RAG", "type": "DISCUSSES"},
]


def seed_graph() -> None:
    """Replace the current database contents with the generic demo graph."""
    print("  Clearing existing graph...")
    clear_graph()
    print(f"  Writing {len(ENTITIES)} entities and {len(RELATIONSHIPS)} relationships...")
    write_graph(ENTITIES, RELATIONSHIPS)
    print("  Graph seeded successfully.")
