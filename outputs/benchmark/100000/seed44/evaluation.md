# BFS vs GNN vs validation-calibrated BFS+GNN

- Held-out test queries: 128
- Validation queries: 128
- Candidate counts (min/median/max): {'min': 6, 'median': 1179.5, 'max': 1377}
- Fusion alpha (GNN weight): **0.55**
- Calibration objective: validation_mrr@10
- BFS fallback selected: False

## TEST (never used to tune alpha)

| Method | K | Hits@K | Recall@K | MRR@K |
|---|---:|---:|---:|---:|
| gnn_only | 1 | 0.0625 | 0.0475 | 0.0625 |
| gnn_only | 3 | 0.1172 | 0.0812 | 0.0872 |
| gnn_only | 5 | 0.1484 | 0.0965 | 0.0939 |
| gnn_only | 10 | 0.2891 | 0.2056 | 0.1115 |
| bfs_only | 1 | 0.0547 | 0.0449 | 0.0547 |
| bfs_only | 3 | 0.1172 | 0.0965 | 0.0833 |
| bfs_only | 5 | 0.1641 | 0.1365 | 0.0939 |
| bfs_only | 10 | 0.2109 | 0.1798 | 0.1007 |
| bfs_gnn | 1 | 0.0703 | 0.0508 | 0.0703 |
| bfs_gnn | 3 | 0.1562 | 0.1097 | 0.1055 |
| bfs_gnn | 5 | 0.2109 | 0.1559 | 0.1180 |
| bfs_gnn | 10 | 0.2891 | 0.2321 | 0.1287 |

## VALIDATION (used for weight selection)

| Method | MRR@10 | Hits@1 |
|---|---:|---:|
| gnn_only | 0.1129 | 0.0547 |
| bfs_only | 0.0779 | 0.0391 |
| bfs_gnn | 0.1383 | 0.0781 |

## Interpretation and safeguards

- All rankings and alpha selection obey the original v6 method.
- This is a transductive held-out link task, not clinical validation.
- Sampling from full graph precedes train/test split.
- Only validation was used to select alpha; no test tuning.
- No artificial cluster-label feature added.
- Reduced Python overhead for 100k triples; semantically equivalent to v6.
