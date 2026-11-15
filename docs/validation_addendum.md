# Validation status for public-data / local-LLM extension

| Item | Result | Evidence |
|---|---|---|
| New Python source and tests syntax | PASS | `compileall` |
| Hetionet TSV / gzipped SIF 3-column fixture, sampled to compact CSV | PASS (fixture using a minimal stand-in for Polars CSV writing) | 30 fictional nodes, 55 fictional relationships → 18 nodes/27 relationships |
| Grounded two-hop traversal and edge-label citations | PASS (isolated stub for Voyager and Polars) | Known A→B→C only; output cites E1/E2 from provided edges |
| Ollama HTTP JSON payload, local-only URL restriction | PASS (mock HTTP response) | `/api/chat`, `stream=false`, `think=false` |
| Actual Polars/Voyager runtime with JAX end-to-end | NOT RUN | Missing dependencies, offline pip |
| Actual Hetionet SIF ingestion | NOT RUN | Actual graph not downloaded/bundled |
| Actual Qwen3/Ollama generation | NOT RUN | Model/server unavailable |
| Full pytest suite | NOT RUN | Cannot import Polars/Voyager |

This is **not** a fully integrated-tested product. Before any live code review,
run `pytest -q` and the two `python -m gnn_rag.rag` commands on the target Mac.
Do not claim accuracy, retrieval speed, RAG relevance, medical validity, or
model quality without an evaluation set and controlled baselines.
