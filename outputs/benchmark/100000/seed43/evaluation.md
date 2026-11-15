# BFS vs GNN vs validation-calibrated BFS+GNN

- Held-out test queries: 128
- Validation queries: 128
- Candidate counts (min/median/max): {'min': 6, 'median': 871.5, 'max': 1377}
- Fusion alpha (GNN weight): **0.80**
- Calibration objective: validation_mrr@10
- BFS fallback selected: False

## TEST (never used to tune alpha)

| Method | K | Hits@K | Recall@K | MRR@K |
|---|---:|---:|---:|---:|
| gnn_only | 1 | 0.0781 | 0.0338 | 0.0781 |
| gnn_only | 3 | 0.1562 | 0.0832 | 0.1068 |
| gnn_only | 5 | 0.2031 | 0.1246 | 0.1177 |
| gnn_only | 10 | 0.2969 | 0.1987 | 0.1306 |
| bfs_only | 1 | 0.0547 | 0.0425 | 0.0547 |
| bfs_only | 3 | 0.1250 | 0.0907 | 0.0833 |
| bfs_only | 5 | 0.1562 | 0.1205 | 0.0904 |
| bfs_only | 10 | 0.2656 | 0.2060 | 0.1055 |
| bfs_gnn | 1 | 0.0469 | 0.0332 | 0.0469 |
| bfs_gnn | 3 | 0.1484 | 0.0913 | 0.0885 |
| bfs_gnn | 5 | 0.2344 | 0.1567 | 0.1081 |
| bfs_gnn | 10 | 0.2812 | 0.1894 | 0.1139 |

## VALIDATION (used for weight selection)

| Method | MRR@10 | Hits@1 |
|---|---:|---:|
| gnn_only | 0.1074 | 0.0234 |
| bfs_only | 0.1257 | 0.0781 |
| bfs_gnn | 0.1633 | 0.0938 |

## Interpretation and safeguards

- All rankings and alpha selection obey the original v6 method.
- This is a transductive held-out link task, not clinical validation.
- Sampling from full graph precedes train/test split.
- Only validation was used to select alpha; no test tuning.
- No artificial cluster-label feature added.
- Reduced Python overhead for 100k triples; semantically equivalent to v6.
