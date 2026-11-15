# Hetionet Full-source → 10k Triple Benchmark

## Result status

**Hetionet actual evaluation: NOT RUN in the artifact-authoring environment.** The full Hetionet raw data could not be downloaded in this environment. No Hetionet MRR/Hits score is supplied or inferred from synthetic data.

### Pending metrics table (fill only from outputs/local_rag/evaluation.json)

| Method | Test Hits@1 | Test Hits@10 | Test MRR@10 |
|---|---:|---:|---:|
| GNN | — | — | — |
| BFS | — | — | — |
| BFS+GNN | — | — | — |

The validation-tuned alpha, candidate-set sizes, relation frequency and per-query rows must accompany a reported result. A hybrid that selects α=0 is BFS fallback, not a demonstrated improvement.

## How to generate real metrics

```bash
cp .env.example .env
mkdir -p data/raw outputs
docker compose up --build prepare-data train
cat data/hetionet_demo/dataset_meta.json
cat outputs/local_rag/evaluation.md
```

Full benchmarking protocol and risks: `README.md` and `docs/v6_validation_status.md`.
