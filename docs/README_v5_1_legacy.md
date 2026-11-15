# GNN-RAG v5.1 — Hidden-cluster, no-clan ablation (GNN vs BFS vs fusion)

**Scope:** relation-conditioned link prediction on typed `(head, relation, tail)` triples, with an optional local Qwen3 RAG demonstration. The three-way evaluation uses **one training graph, one candidate set per query, and one held-out test split**. Hyperparameter selection uses **validation only**. Keep JAX / Polars / Voyager / Docker Compose / Ollama.

> This is a research/demo implementation. Predicted missing KG links are hypotheses, not verified medical evidence. Only *observed training edges* may be rendered as LLM evidence.

## Revised synthetic evaluation: no clan information (2026-11-14)

The **default** offline synthetic experiment is now **strict**: node names have only two values (`Synthetic disease`, `Synthetic gene`), GraphSAGE receives only a 2-dimensional **node-type one-hot** (no names or node IDs), and synthetic node indices are randomly remapped within types to eliminate the `index % 6` cue and avoid index-based tie-breaking bias. The topology still contains deliberately generated latent communities. **The latent community IDs are never passed to the model or the ranking function.**

| Experiment | Seeds | GNN MRR@10 | BFS MRR@10 | Fusion MRR@10 |
|---|---:|---:|---:|---:|
| **Strict: type-only + scrambled IDs** | 41–43 | 0.0532 | **0.1487** | 0.1435 |
| Paired feature ablation: type-only + original IDs | 41–43 | 0.1006 | 0.1262 | **0.1689** |
| Sensitivity: type-only + original IDs | 41–50 | 0.0794 | 0.1461 | **0.1744** |
| **Previous v5 (cluster-encoded node names)** | 41–43 | **0.3847** | 0.1262 | 0.3847 |

**Interpretation:** the previous GNN success does not generalize to removing synthetic cluster hints. In the strictest condition, **BFS outperforms both GNN and the validation-tuned hybrid** on the three measured seeds. The paired ablation can show an apparent hybrid gain, but the gain does not hold in the index-scrambled control; it must not be claimed as robust. No Hetionet data were evaluated in the offline synthetic experiment.

Reproduce (requires JAX and NumPy only):

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/offline_compare.py --seeds 41 42 43 --epochs 25 --output examples/reproduced_strict.json
# optional: same graph/index scheme as the old experiment, but remove clan features
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/offline_compare.py --no-scramble-indices --seeds 41 42 43 --epochs 25 --output examples/reproduced_paired.json
```

Full numbers, split details, failure modes, and raw JSON/Markdown outputs: [docs/v5_1_no_clan_evaluation.md](docs/v5_1_no_clan_evaluation.md). **Docker Compose/Hetionet pipeline remains unchanged and uses original real-world name features by default**; the no-clan ablation is for the synthetic benchmark only.

## What changed from v4

| Method | Ranking definition | Where it gets its information |
|---|---|---|
| `gnn_only` | Exact ranking of `cosine(normalize(z_h + u_r), z_t)` | JAX GraphSAGE+relation vectors trained on training triples |
| `bfs_only` | Ascending shortest-path distance; ties by descending degree then node ID | Undirected adjacency built from **training triples only** |
| `bfs_gnn` | Validation-calibrated combination of BFS rank percentile and GNN rank percentile | Training-only GNN/BFS plus a **single global alpha tuned on validation** |

For query `(head, relation)` and each eligible tail `t`, let `rank_G(t)`, `rank_B(t)` be 0-based positions among the *same* candidate tails. With `n` candidates:

```text
P_G(t) = 1 - rank_G(t) / max(n-1, 1)
P_B(t) = 1 - rank_B(t) / max(n-1, 1)
S_alpha(t) = alpha * P_G(t) + (1-alpha) * P_B(t)
```

The alpha grid is `0.00, 0.05, ..., 1.00`, chosen by **validation macro MRR@10**; ties favor Hits@1 and then smaller alpha. `alpha=0` *exactly* reproduces BFS; `alpha=1` *exactly* reproduces GNN. A chosen alpha of zero **is BFS fallback** and is reported as such, not as a gain. A near-one alpha does not imply a fusion improvement.

**Important:** this measures triple recovery, not a biomedical causal relation or final answer accuracy. Unlike v3.1, candidates are relation/type constrained. Do NOT compare old v3.1 numerical scores directly with the v5 benchmark.

## Mac / Docker Compose

Unzip into a **new folder**. Do not copy old v3/v4 `outputs/` artifacts into this project. If the old project still occupies port 8000, stop it from its own project directory first (`docker compose down`).

```bash
cp .env.example .env
# .env contains GNN_RAG_DATASET=hetionet by default
# Optional: copy previously verified Hetionet raw files into ./data/raw/
# cp -R ../GNN-RAG_v2/data/raw/. ./data/raw/

docker compose --profile test run --build --rm test
docker compose up --build -d --force-recreate
docker compose ps -a
docker compose logs --no-color --tail=120 prepare-data train api

cat outputs/local_rag/evaluation.md
curl -fsS http://127.0.0.1:8000/evaluation
```

`prepare-data` downloads Hetionet once if missing; cached validated files are reused, the preprocessing conserves triples (schema v4). `train` completes and exits 0; Ollama/API stay up. Results are in `outputs/local_rag/`.

The report files:

- `evaluation.md`: **GNN-only / BFS-only / BFS+GNN** validation/test comparison
- `evaluation.json`: full metrics, alpha calibration grid, candidate sizes, evaluation boundaries
- `evaluation_per_query.csv`: test rows for all 3 methods, matched by `(source, relation)`
- `train_edges.csv`: observed training-only triples for verification and API search
- `run_summary.json`, `embeddings.npy`, `relation_vectors.npy`: trained artifacts

### Triple predictions via API

```bash
curl -fsS -X POST http://127.0.0.1:8000/predict-triples \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"Disease::DOID:10652","relation":"DrD","top_k":5,"method":"bfs_gnn"}'
```

`method` can be `gnn_only`, `bfs_only`, `bfs_gnn`. The API loads the validation-calibrated alpha from `evaluation.json`; it never tunes on a user request. **The reported `gnn_score` is a component score, not the blended rank.** The API response explicitly warns that missing-link predictions are not observed facts.

Existing `/evidence` and `/answer` are not changed to treat predicted links as facts. Voyager continues to support the existing ANN/API pipeline but is *not* substituted for exact GNN scoring in the three-method correctness benchmark (this avoids confounding ANN recall with ranking quality).

### Reproduce the offline comparison without Hetionet / Polars / Voyager

```bash
python -m pip install 'jax>=0.4.30' numpy pytest
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/offline_compare.py --seeds 41 42 43 --epochs 25
cat examples/offline_benchmark_strict.md
PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_v5_fusion.py tests/test_v4_triple_core.py
```

The bundled `examples/offline_benchmark_no_clan_scrambled_3seeds.{md,json}` are generated with JAX on the strict no-clan benchmark. Previous cluster-hinted results are archived under `examples/legacy_v5_with_clan/` for audit, and must not be used as valid evidence of model superiority. All synthetic metrics are **not Hetionet metrics**. See `docs/v5_1_no_clan_evaluation.md`.

### Interpreting results

1. First inspect `calibration.alpha`, `fallback_to_bfs`, and the validation score. If alpha is zero, the hybrid falls back to BFS.
2. Compare the three `test` metrics **on exactly the same split**, especially MRR@3, MRR@10, and Hits@1; Hits@10 may be saturated for tiny relation-wise candidate sets.
3. Check `candidate_counts_test` and per-query CSV before any claim about model superiority.
4. Repeat multiple random seeds and preferably use a larger, less centrally sampled Hetionet subgraph; the one-split results are descriptive, not statistical significance.

For benchmark design and actual offline results see [v5 comparison report](docs/v5_comparison.md). Previous v4 background remains in [archived README](docs/README_v4_legacy.md).

**Licensing:** see `LICENSE-NOTE.md`. Raw Hetionet and Qwen3 model weights are not included in this ZIP.

### Optional: independent three-seed Hetionet evaluation (recommended)

For stronger evidence than one split, after normal Compose setup (or with automatic `prepare-data` as a dependency), run:

```bash
docker compose --profile benchmark run --build --rm benchmark
cat outputs/multiseed/summary.md
```

This separately trains/evaluates seeds **41, 42 and 43**, selecting the fusion weight on validation *within each seed*, and writes per-seed outputs under `outputs/multiseed/seed_*/` plus an aggregate Markdown/JSON report. It does **not** overwrite `outputs/local_rag/`, which serves the API. The reported sample std is descriptive, not a significance test. This optional job requires Hetionet CSVs (`.env` defaults to Hetionet), not the synthetic mode.
