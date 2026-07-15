# GraphMind User Guide

## 1. Install

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure the API and Neo4j values in `.env`.

## 2. Add authorized sources

Place documents beneath `data/raw/`. A top-level directory becomes its collection name:

```text
data/raw/
├── engineering/
│   ├── architecture.docx
│   └── services.xlsx
└── research/
    ├── paper.pdf
    └── notes.md
```

Supported formats are text/Markdown, JSON/JSONL, CSV/TSV, PDF, DOCX, XLSX/XLSM, and HTML.

## 3. Run incremental ingestion

```powershell
python scripts\preprocess_raw_data.py
```

The command reports discovered, parsed, unchanged, failed, and removed files. Parsing errors are recorded under `data/processed/ingestion_errors.json` without stopping unrelated sources.

Use `--force` to reparse unchanged files or `--collection NAME` to set a collection explicitly.

## 4. Build indexes

Build and load the graph:

```powershell
python scripts\extract_graph_from_processed.py --write-neo4j --clear
```

Build the semantic index:

```powershell
python -c "from src.retrieval.vector_rag import build_vector_store; build_vector_store()"
```

BM25, metadata, and source inspection read the canonical chunks directly and do not require a separate build step.

## 5. Ask questions

```powershell
streamlit run src\ui\app.py
```

The main assistant displays its retrieval plan, tool calls, verification confidence, retries, and sources. The comparison view runs GraphRAG and vector RAG separately.

## 6. Customize the ontology

Copy `config/default_ontology.json`, edit its labels and relationships, and set:

```text
ONTOLOGY_PATH=config/my_ontology.json
```

`Document`, `Chunk`, `PART_OF`, and `DISCUSSES` are required structural elements. Other entity and relationship types can be changed for a domain.

## 7. Connect an MCP client

Start the server with:

```powershell
python -m src.mcp_server
```

Use `mcp_config.example.json` as the client configuration template. Start with `ask_graphmind` and `knowledge_stats`.

## 8. Validate

```powershell
python -m unittest discover -s tests -v
python scripts\smoke_test_mcp.py --question "Which documents discuss vector search?"
```

## Data safety

Never commit `.env`, `data/`, `chroma_db/`, logs, database dumps, or source documents. These locations are excluded by the repository `.gitignore`.
