# GraphMind

GraphMind is a connected knowledge assistant that combines GraphRAG and vector search. It indexes documents, builds relationships between entities, and routes each question to the retrieval strategy best suited to answer it.

## Why GraphMind

Vector search is useful for summaries and source-grounded explanations. A knowledge graph is better at answering relationship questions such as which concepts, tools, documents, or people are connected. GraphMind combines both approaches behind one interface.

## Features

- Neo4j knowledge graph for structured relationships
- Chroma vector store for semantic document retrieval
- Hybrid routing with LangChain and LangGraph
- Source-grounded answers with document metadata
- Streamlit interface for exploration and comparison
- MCP server for use from compatible AI clients
- Read-only Cypher tool with mutation safeguards

## Architecture

```text
Documents
   |
   +-- preprocessing --> processed JSONL --> Chroma vector store
   |
   +-- entity extraction --> graph seed --> Neo4j
                                      |
Question --> LangGraph router --------+--> GraphRAG answer
             |                        |
             +------------------------+--> Vector RAG answer
```

## Project Structure

```text
GraphMind/
|-- src/
|   |-- graph/          # Graph extraction and Neo4j access
|   |-- ingestion/      # Loading, cleaning, and preprocessing
|   |-- retrieval/      # Graph, vector, and hybrid retrieval
|   |-- ui/             # Streamlit application
|   |-- utils/          # Configuration and embeddings
|   `-- mcp_server.py   # MCP tools and resources
|-- scripts/            # Processing, evaluation, and smoke tests
|-- .env.example
|-- requirements.txt
`-- run_mcp_server.bat
```

## Setup

### 1. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Set your model provider and Neo4j credentials in `.env`. Never commit that file.

### 4. Add documents

Place source documents under `data/raw/`. The `data/`, `chroma_db/`, and `.env` paths are intentionally ignored so private source material and local indexes are not published.

### 5. Process documents

```powershell
python scripts\preprocess_raw_data.py
```

To build or load the graph:

```powershell
python scripts\extract_graph_from_processed.py --write-neo4j
```

### 6. Run the application

```powershell
streamlit run src\ui\app.py
```

## MCP Server

Start the MCP server directly:

```powershell
python -m src.mcp_server
```

Or on Windows:

```powershell
.\run_mcp_server.bat
```

Available tools:

- `ask_graphmind` — automatically selects graph or vector retrieval
- `search_knowledge_base` — performs semantic source retrieval
- `query_knowledge_graph` — answers structured relationship questions
- `run_readonly_cypher` — runs guarded read-only Cypher
- `knowledge_stats` — reports the current index and graph inventory

Use [mcp_config.example.json](mcp_config.example.json) as a portable configuration example.

## Privacy

GraphMind does not include source documents, processed datasets, local vector stores, credentials, or database exports. Confirm that you have permission to index and share any documents you add.

## License

This project is available under the MIT License. See [LICENSE](LICENSE).
