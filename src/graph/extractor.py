import json
from openai import OpenAI
from src.graph.ontology import DEFAULT_ONTOLOGY
from src.utils.config import NEBIUS_API_KEY, NEBIUS_BASE_URL, LLM_MODEL
from src.utils.entity_resolver import resolve_entities

def _client() -> OpenAI:
    if not NEBIUS_API_KEY:
        raise RuntimeError("NEBIUS_API_KEY is not configured")
    return OpenAI(api_key=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)

ENTITY_TYPES = list(DEFAULT_ONTOLOGY.extraction_labels)
RELATIONSHIP_TYPES = list(DEFAULT_ONTOLOGY.relationship_types)

EXTRACTION_PROMPT = """You are a knowledge graph extractor. Read the text below and extract entities and relationships.

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "entity name", "type": "ENTITY_TYPE"}}
  ],
  "relationships": [
    {{"source": "entity name", "target": "entity name", "type": "RELATIONSHIP_TYPE"}}
  ]
}}

Allowed entity types: {entity_types}
Allowed relationship types (use ONLY these): {relationship_types}

Rules:
- Only extract meaningful named entities (people, organizations, tools, projects, resources, topics, and concepts)
- Do NOT extract generic words like "page 2", "model", "framework", "stack", "use case"
- Every relationship source and target must appear in the entities list
- Use ONLY the allowed relationship types listed above
- If no clear entities or relationships found, return empty lists
- Do not add explanation, only return JSON

Text:
{text}"""


def build_extraction_prompt(chunk_text: str) -> str:
    return EXTRACTION_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES),
        relationship_types=", ".join(RELATIONSHIP_TYPES),
        text=chunk_text
    )


def extract_from_chunk(chunk_text: str) -> dict:
    prompt = build_extraction_prompt(chunk_text)
    response = _client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()

    # strip markdown code fences if model wraps response in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def extract_graph_elements(chunks: list) -> dict:
    all_entities = {}
    all_relationships = []

    for i, chunk in enumerate(chunks):
        print(f"  Extracting chunk {i + 1}/{len(chunks)}...")
        try:
            result = extract_from_chunk(chunk.page_content)

            for entity in result.get("entities", []):
                key = (entity["name"].strip(), entity["type"].strip())
                all_entities[key] = {"name": key[0], "type": key[1]}

            for rel in result.get("relationships", []):
                all_relationships.append({
                    "source": rel["source"].strip(),
                    "target": rel["target"].strip(),
                    "type": rel["type"].strip().upper(),
                })

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: chunk {i + 1} skipped — {e}")
            continue

    raw = {
        "entities": list(all_entities.values()),
        "relationships": all_relationships,
    }
    return resolve_entities(raw)
