# GraphMind User Guide

## 1. Install

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

You can defer `.env` configuration until semantic or graph retrieval is needed.

## 2. Try the bundled demo first

The synthetic HelioDesk corpus covers Markdown, CSV, JSON, HTML, DOCX, XLSX, and PDF. Ingest it and run a BM25 smoke test without credentials or Neo4j:

```powershell
python scripts\bootstrap_demo.py --force
```

## 3. Add authorized sources

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

## 4. Run incremental ingestion

```powershell
python scripts\preprocess_raw_data.py
```

The command reports discovered, parsed, unchanged, failed, and removed files. Parsing errors are recorded under `data/processed/ingestion_errors.json` without stopping unrelated sources.

Use `--force` to reparse unchanged files, `--collection NAME` to set a collection explicitly, or `--raw-dir PATH` to ingest a different source directory.

## 5. Build indexes

Build the semantic index after adding model credentials:

```powershell
python -c "from src.retrieval.vector_rag import build_vector_store; build_vector_store()"
```

Connect Neo4j only when you are ready to add relationship retrieval. Set the Neo4j and model values in `.env`, verify the connection, then build and load the graph:

```powershell
python scripts\check_neo4j.py
python scripts\extract_graph_from_processed.py --write-neo4j --clear
```

BM25, metadata, and source inspection read the canonical chunks directly and do not require a separate build step.

## 6. Ask questions

```powershell
streamlit run src\ui\app.py
```

The main assistant displays its retrieval plan, tool calls, verification confidence, retries, and sources. The comparison view runs GraphRAG and vector RAG separately.

## 7. Customize the ontology

Copy `config/default_ontology.json`, edit its labels and relationships, and set:

```text
ONTOLOGY_PATH=config/my_ontology.json
```

`Document`, `Chunk`, `PART_OF`, and `DISCUSSES` are required structural elements. Other entity and relationship types can be changed for a domain.

## 8. Connect an MCP client

Start the server with:

```powershell
python -m src.mcp_server
```

Use `mcp_config.example.json` as the client configuration template. Start with `ask_graphmind` and `knowledge_stats`.

## 9. Validate

```powershell
python -m unittest discover -s tests -v
python scripts\smoke_test_mcp.py --question "Which documents discuss vector search?"
```

## Data safety

Never commit `.env`, `data/`, `chroma_db/`, logs, database dumps, or private source documents. Only the explicitly synthetic files under `demo_data/` are intended for version control.
