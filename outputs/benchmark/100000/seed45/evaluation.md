# BFS vs GNN vs validation-calibrated BFS+GNN

- Held-out test queries: 128
- Validation queries: 128
- Candidate counts (min/median/max): {'min': 6, 'median': 923.0, 'max': 1377}
- Fusion alpha (GNN weight): **0.90**
- Calibration objective: validation_mrr@10
- BFS fallback selected: False

## TEST (never used to tune alpha)

| Method | K | Hits@K | Recall@K | MRR@K |
|---|---:|---:|---:|---:|
| gnn_only | 1 | 0.0469 | 0.0352 | 0.0469 |
| gnn_only | 3 | 0.1016 | 0.0513 | 0.0716 |
| gnn_only | 5 | 0.1875 | 0.1210 | 0.0904 |
| gnn_only | 10 | 0.3125 | 0.2047 | 0.1061 |
| bfs_only | 1 | 0.0547 | 0.0358 | 0.0547 |
| bfs_only | 3 | 0.1406 | 0.0828 | 0.0898 |
| bfs_only | 5 | 0.1641 | 0.1079 | 0.0953 |
| bfs_only | 10 | 0.2891 | 0.2319 | 0.1118 |
| bfs_gnn | 1 | 0.0625 | 0.0329 | 0.0625 |
| bfs_gnn | 3 | 0.1641 | 0.0833 | 0.1055 |
| bfs_gnn | 5 | 0.2422 | 0.1586 | 0.1230 |
| bfs_gnn | 10 | 0.3203 | 0.2338 | 0.1332 |

## VALIDATION (used for weight selection)

| Method | MRR@10 | Hits@1 |
|---|---:|---:|
| gnn_only | 0.1088 | 0.0547 |
| bfs_only | 0.0757 | 0.0312 |
| bfs_gnn | 0.1215 | 0.0625 |

## Interpretation and safeguards

- All rankings and alpha selection obey the original v6 method.
- This is a transductive held-out link task, not clinical validation.
- Sampling from full graph precedes train/test split.
- Only validation was used to select alpha; no test tuning.
- No artificial cluster-label feature added.
- Reduced Python overhead for 100k triples; semantically equivalent to v6.
