# BFS vs GNN vs validation-calibrated BFS+GNN

- Held-out test queries: 128
- Validation queries: 128
- Candidate counts (min/median/max): {'min': 7, 'median': 170.0, 'max': 270}
- Fusion alpha (GNN weight): **0.95**
- Calibration objective: validation_mrr@10
- BFS fallback selected: False

## TEST (never used to tune alpha)

| Method | K | Hits@K | Recall@K | MRR@K |
|---|---:|---:|---:|---:|
| gnn_only | 1 | 0.1484 | 0.1319 | 0.1484 |
| gnn_only | 3 | 0.3281 | 0.2826 | 0.2279 |
| gnn_only | 5 | 0.4141 | 0.3475 | 0.2462 |
| gnn_only | 10 | 0.5781 | 0.4685 | 0.2682 |
| bfs_only | 1 | 0.1406 | 0.1276 | 0.1406 |
| bfs_only | 3 | 0.2812 | 0.2562 | 0.1992 |
| bfs_only | 5 | 0.3594 | 0.3356 | 0.2168 |
| bfs_only | 10 | 0.4297 | 0.4070 | 0.2264 |
| bfs_gnn | 1 | 0.1562 | 0.1419 | 0.1562 |
| bfs_gnn | 3 | 0.3125 | 0.2811 | 0.2253 |
| bfs_gnn | 5 | 0.3984 | 0.3468 | 0.2444 |
| bfs_gnn | 10 | 0.5625 | 0.4574 | 0.2671 |

## VALIDATION (used for weight selection)

| Method | MRR@10 | Hits@1 |
|---|---:|---:|
| gnn_only | 0.2457 | 0.1484 |
| bfs_only | 0.1818 | 0.1094 |
| bfs_gnn | 0.2459 | 0.1562 |

## Interpretation and safeguards

- All rankings and alpha selection obey the original v6 method.
- This is a transductive held-out link task, not clinical validation.
- Sampling from full graph precedes train/test split.
- Only validation was used to select alpha; no test tuning.
- No artificial cluster-label feature added.
- Reduced Python overhead for 100k triples; semantically equivalent to v6.
