# BFS vs GNN vs validation-calibrated BFS+GNN

- Held-out test queries: 128
- Validation queries: 128
- Candidate counts (min/median/max): {'min': 7, 'median': 942.0, 'max': 1377}
- Fusion alpha (GNN weight): **0.85**
- Calibration objective: validation_mrr@10
- BFS fallback selected: False

## TEST (never used to tune alpha)

| Method | K | Hits@K | Recall@K | MRR@K |
|---|---:|---:|---:|---:|
| gnn_only | 1 | 0.0625 | 0.0431 | 0.0625 |
| gnn_only | 3 | 0.1797 | 0.0997 | 0.1146 |
| gnn_only | 5 | 0.2422 | 0.1418 | 0.1286 |
| gnn_only | 10 | 0.3906 | 0.2448 | 0.1497 |
| bfs_only | 1 | 0.0469 | 0.0294 | 0.0469 |
| bfs_only | 3 | 0.1328 | 0.0998 | 0.0820 |
| bfs_only | 5 | 0.2109 | 0.1553 | 0.0992 |
| bfs_only | 10 | 0.2969 | 0.2368 | 0.1095 |
| bfs_gnn | 1 | 0.1016 | 0.0755 | 0.1016 |
| bfs_gnn | 3 | 0.1797 | 0.1388 | 0.1341 |
| bfs_gnn | 5 | 0.2031 | 0.1624 | 0.1400 |
| bfs_gnn | 10 | 0.3203 | 0.2347 | 0.1555 |

## VALIDATION (used for weight selection)

| Method | MRR@10 | Hits@1 |
|---|---:|---:|
| gnn_only | 0.1194 | 0.0312 |
| bfs_only | 0.1052 | 0.0391 |
| bfs_gnn | 0.1428 | 0.0703 |

## Interpretation and safeguards

- All rankings and alpha selection obey the original v6 method.
- This is a transductive held-out link task, not clinical validation.
- Sampling from full graph precedes train/test split.
- Only validation was used to select alpha; no test tuning.
- No artificial cluster-label feature added.
- Reduced Python overhead for 100k triples; semantically equivalent to v6.
