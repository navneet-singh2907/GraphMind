# HelioDesk synthetic demo corpus

This directory contains a fully fictional, redistributable corpus for demonstrating GraphMind. HelioDesk, Asteria Systems, its people, incidents, prices, metrics, services, and customer scenarios are invented for this repository. They do not describe a real company or product.

The corpus deliberately repeats and connects facts across formats so retrieval can be tested on exact lookup, synthesis, source inspection, and relationship questions.

| File | Format | Main evidence |
|---|---|---|
| `sources/engineering/architecture.md` | Markdown | components, dependencies, data flow, controls |
| `sources/operations/service_catalog.csv` | CSV | owners, SLOs, recovery targets, regions |
| `sources/product/product_spec.json` | JSON | plans, capabilities, integrations, limits |
| `sources/support/customer_faq.html` | HTML | customer-facing behavior and escalation rules |
| `sources/strategy/implementation_guide.docx` | DOCX | phased rollout, roles, acceptance criteria |
| `sources/finance/pricing_capacity_model.xlsx` | XLSX | auditable pricing and capacity formulas |
| `sources/reliability/q2_reliability_review.pdf` | PDF | metrics, incident timeline, recommendations |

## Run without Neo4j

From the repository root:

```powershell
python scripts\bootstrap_demo.py --force
```

This parses every format into `data/processed/document_chunks.jsonl` and runs a local BM25 keyword smoke test. It does not need Neo4j or model credentials.

## When to connect Neo4j

Do not connect it for the first ingestion test. First confirm that the command above discovers seven files with zero failures. Add model credentials next if you want semantic embeddings. Connect Neo4j only for the final graph layer:

```powershell
python scripts\check_neo4j.py
python scripts\extract_graph_from_processed.py --write-neo4j --clear
```

The second command extracts relationships from the canonical chunks and writes them to Neo4j. If Neo4j is later unavailable, GraphMind can still use keyword and vector retrieval.

Use of the corpus is covered by the repository's MIT license.
