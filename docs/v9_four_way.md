# v9: Qwen only / BFS only / GNN only / BFS+GNN + four-way PDF

## Scope

The four outputs are **answers to the same free-text question in Graph Explorer**,
not four quantitative held-out-link metrics. The existing 10k/100k benchmark is
unchanged: its MRR@10 / Hits@10 / Recall@10 apply to BFS/GNN/fusion ranking only.
Raw Qwen has no comparable link-ranking score and is explicitly marked unverified.

### Implementation

- `POST /compare-answers`: requires an existing training run / observed train edges
- **Qwen only**: same user question plus the selected node's name, with **no KG context**;
  the Ollama model generates free text (`think:false`, deterministic temperature 0)
- **BFS only**: candidate paths restricted to TRAIN edges; expansion is BFS,
  with equal-depth ties ranked by observed neighbor degree and ID
- **GNN only**: identical observed candidate paths; outgoing edges ranked by
  trained relation-conditioned GraphSAGE score
- **BFS + GNN**: combination of local BFS/GNN rank percentiles using the `alpha`
  from saved **validation** calibration. If no valid alpha exists, fallback is
  `0.5` and the interface reports it as uncalibrated
- For each KG method, **Qwen selects valid evidence IDs only** under a JSON
  schema, then a deterministic KG renderer writes the final text. No ungrounded
  medical prose from the model is included in the KG method outputs
- All candidate paths, facts and selections are available on the UI and saved
  in the JSON. The PDF contains all four actual answers and KG evidence.

This is a qualitative demonstration, not a controlled medical QA benchmark.
No quantitative truthfulness scores, hallucination rates or clinical accuracy
are claimed. The 10k/100k strict held-out rank evaluation and its PDF still use
only the three graph ranking methods.

## Commands

```bash
[ -f .env ] || cp .env.example .env
# If updating an old .env: HETIONET_MAX_NODES=5000
# and GNN_RAG_DATASET=hetionet
docker compose --profile test run --build --rm test
docker compose up --build -d
# Open http://127.0.0.1:5173/ -> Graph Explorer
# Choose a node, type a question, then click "4方式を比較してPDFに保存"
ls -lh GNN_RAG_four_way_answers.pdf GNN_RAG_four_way_answers.json
open GNN_RAG_four_way_answers.pdf
# Separate *quantitative* held-out rank benchmark:
docker compose --profile study run --build --rm study
ls -lh Hetionet_10k_100k_comparison.pdf
```

API equivalents:

```bash
curl -sS -X POST http://127.0.0.1:8000/compare-answers \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"Disease::DOID:10652","question":"関連する概念は何ですか？","hops":2,"top_k":3}'
curl -fSL http://127.0.0.1:8000/comparison-report -o GNN_RAG_four_way_answers_download.pdf
```

With the React frontend, endpoints are proxied as `/api/compare-answers`
and `/api/comparison-report`. The API writes fixed-name PDF/JSON files to the
host project root mounted at `/app/project_root`. **The last run replaces the
previous comparison report**, so copy them elsewhere if keeping experiments.

Security: this is a localhost-only research UI. Do not enter patient data.
The app's mounted root folder is writable by the local API container to save
PDF/JSON, so do not expose it publicly. `GNN_RAG_UI_PORT` sets the UI port.

## Smoke / troubleshooting

```bash
docker compose ps -a
docker compose logs --tail=100 api ollama frontend
curl -sS http://127.0.0.1:8000/health
```

If Qwen isn't available, each method explicitly records `error`, not a fabricated
answer. A generated PDF can therefore include failed methods with diagnostic
statuses rather than pretending that all four calls succeeded. Calls are
sequential on a CPU-only Ollama container; this comparison may take time.
