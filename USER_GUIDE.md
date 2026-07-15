# GraphMind User Guide

GraphMind turns a local document collection into a searchable vector index and connected Neo4j knowledge graph.

## Prerequisites

- Python 3.11 or later
- A running Neo4j database
- Model-provider credentials configured in `.env`
- Source documents you are authorized to process

## Installation

From the project directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set:

- `NEBIUS_API_KEY`
- `NEBIUS_BASE_URL`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- the desired model names

## Prepare the Knowledge Base

Put authorized documents beneath `data/raw/`, then run:

```powershell
python scripts\preprocess_raw_data.py
```

Build the extracted graph and write it to Neo4j:

```powershell
python scripts\extract_graph_from_processed.py --write-neo4j
```

Generated data and local indexes are ignored by Git.

## Run the Streamlit Interface

```powershell
streamlit run src\ui\app.py
```

Use the hybrid assistant for normal questions. The comparison view shows how graph retrieval and semantic retrieval approach the same question differently.

## Connect an MCP Client

The server runs over standard input/output:

```powershell
python -m src.mcp_server
```

Copy `mcp_config.example.json`, replace the placeholder project path, and add it to your MCP client's configuration.

GraphMind exposes:

- `ask_graphmind`
- `search_knowledge_base`
- `query_knowledge_graph`
- `run_readonly_cypher`
- `knowledge_stats`

## Smoke Test

```powershell
python scripts\smoke_test_mcp.py --question "Which concepts are connected to GraphRAG?"
```

## Troubleshooting

If Neo4j cannot be reached, verify the service is running and that the URI and credentials in `.env` are correct.

If vector search returns no results, confirm that processed vector documents exist and rebuild the local Chroma store.

If an MCP client does not display GraphMind, run the server command manually and check that the configuration uses absolute Windows paths where required.

## Data Safety

Do not commit `.env`, raw documents, processed document exports, database dumps, or `chroma_db`. The provided `.gitignore` excludes these paths by default.
