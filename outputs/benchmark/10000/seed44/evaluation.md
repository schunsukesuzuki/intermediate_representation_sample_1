# BFS vs GNN vs validation-calibrated BFS+GNN

- Held-out test queries: 128
- Validation queries: 128
- Candidate counts (min/median/max): {'min': 1, 'median': 220.5, 'max': 270}
- Fusion alpha (GNN weight): **0.70**
- Calibration objective: validation_mrr@10
- BFS fallback selected: False

## TEST (never used to tune alpha)

| Method | K | Hits@K | Recall@K | MRR@K |
|---|---:|---:|---:|---:|
| gnn_only | 1 | 0.1562 | 0.1220 | 0.1562 |
| gnn_only | 3 | 0.3047 | 0.2377 | 0.2135 |
| gnn_only | 5 | 0.3906 | 0.3067 | 0.2342 |
| gnn_only | 10 | 0.5234 | 0.4268 | 0.2521 |
| bfs_only | 1 | 0.0938 | 0.0824 | 0.0938 |
| bfs_only | 3 | 0.2109 | 0.2000 | 0.1471 |
| bfs_only | 5 | 0.2891 | 0.2768 | 0.1643 |
| bfs_only | 10 | 0.4062 | 0.3850 | 0.1805 |
| bfs_gnn | 1 | 0.1719 | 0.1374 | 0.1719 |
| bfs_gnn | 3 | 0.3125 | 0.2884 | 0.2305 |
| bfs_gnn | 5 | 0.3672 | 0.3289 | 0.2434 |
| bfs_gnn | 10 | 0.5078 | 0.4450 | 0.2626 |

## VALIDATION (used for weight selection)

| Method | MRR@10 | Hits@1 |
|---|---:|---:|
| gnn_only | 0.2376 | 0.1172 |
| bfs_only | 0.1877 | 0.0703 |
| bfs_gnn | 0.2587 | 0.1719 |

## Interpretation and safeguards

- All rankings and alpha selection obey the original v6 method.
- This is a transductive held-out link task, not clinical validation.
- Sampling from full graph precedes train/test split.
- Only validation was used to select alpha; no test tuning.
- No artificial cluster-label feature added.
- Reduced Python overhead for 100k triples; semantically equivalent to v6.
