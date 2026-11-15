# v7 Hetionet 10k / 100k: implementation and validation

## What changed

- `dataset_archive.py`: source resolution: verified local TSV+SIF.gz → supplied archive (`data/archives/hetionet-v1.0.0.zip`) → official HTTPS; ZIP extraction is limited to two expected members and uses temporary files validated before replacing cache.
- `dual_scale_benchmark.py`: independent samples at 10,000 and 100,000 typed triples, fixed sample seed 42, repeated JAX training at seeds 42–46; same-source full Hetionet population.
- Single-scale comparison uses the pre-existing v6 GraphSAGE encoder, relation-conditioned score, TRAIN-only BFS, validation-tuned rank-percentile fusion, and full-candidate exact ranking. No new heuristics or clusters are injected.
- Outputs `overview.md`, `overview.json`, `summary.csv`, per-scale summaries, and per-seed JSON/Markdown/CSV/embeddings. The API adds read-only `/benchmark-summary`.
- Stable Polars source/target edge ordering, matching previous pandas verification adapter, was made explicit.

## Verification in the build environment (2026-11-14)

1. `tests/test_dual_scale_study.py`: 4/4 passed; tests CLI defaults, missing inputs with downloads disabled, invalid archives, metrics/Markdown/JSON/CSV aggregation.
2. The actual attached `hetionet-v1.0.0.zip` (148 MB) was read without network; source SIF.gz passed SHA-256; extracted 10,000 unique triples (495 nodes, 24 relations) and 100,000 unique triples (2,330 nodes, 24 relations).
3. Real Hetionet sample 10k: JAX training+three-method evaluation+reports completed with 2 epochs, positive cap 512, 16 test queries (smoke test ONLY). 100k: completed with 1 epoch, same cap and query limit (smoke test ONLY).
4. Local tests used a **pandas CSV adapter only** because Polars was unavailable in this execution environment. Core GraphSAGE/JAX training and exact candidate ranking were the same code used by Docker. Docker itself was unavailable; a Mac end-to-end Compose run has **not** been verified for v7.
5. Pre-existing full 25-epoch/5-seed Hetionet measurements are kept separate as reference, not falsely presented as newly produced v7 results.

## Cautions

- 10k and 100k independently selected subgraphs have different entity/candidate distributions. This is **not** a controlled sample-size-only learning curve.
- The full source graph is used for structural sampling *before* train/test splits. All GNN adjacency and BFS distances during training/evaluation use TRAIN edges only. `triple_split` keeps unordered node pairs in the same partition.
- Source labels have no clinical ground truth guarantee. Unobserved relations are not certified negatives. Holdout ranking does not measure the factual correctness of local Qwen3 answer text.
- `train_max_positives=4096`: each epoch's positive ranking loss sees at most this many sampled training triples; all Train triples enter message-passing adjacency.
- Exact ranking intentionally does not call Voyager: it separates approximate indexing effects from model quality. The live API still uses Voyager where applicable.
