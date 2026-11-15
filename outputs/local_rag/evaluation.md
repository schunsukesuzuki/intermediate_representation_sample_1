# BFS vs GNN vs validation-calibrated BFS+GNN

- Held-out test queries: 128
- Validation queries: 128
- Candidate counts (min/median/max): {'min': 2, 'median': 218.0, 'max': 270}
- Fusion alpha (GNN weight): **0.50**
- Calibration objective: validation_mrr@10
- BFS fallback selected: False

## TEST (never used to tune alpha)

| Method | K | Hits@K | Recall@K | MRR@K |
|---|---:|---:|---:|---:|
| gnn_only | 1 | 0.1797 | 0.1432 | 0.1797 |
| gnn_only | 3 | 0.2891 | 0.2182 | 0.2227 |
| gnn_only | 5 | 0.3828 | 0.3063 | 0.2441 |
| gnn_only | 10 | 0.5547 | 0.4607 | 0.2680 |
| bfs_only | 1 | 0.1172 | 0.1133 | 0.1172 |
| bfs_only | 3 | 0.2578 | 0.2198 | 0.1810 |
| bfs_only | 5 | 0.3359 | 0.3052 | 0.1986 |
| bfs_only | 10 | 0.4141 | 0.3882 | 0.2094 |
| bfs_gnn | 1 | 0.1719 | 0.1475 | 0.1719 |
| bfs_gnn | 3 | 0.2969 | 0.2654 | 0.2279 |
| bfs_gnn | 5 | 0.3984 | 0.3370 | 0.2501 |
| bfs_gnn | 10 | 0.5000 | 0.4335 | 0.2636 |

## VALIDATION (used for weight selection)

| Method | MRR@10 | Hits@1 |
|---|---:|---:|
| gnn_only | 0.2224 | 0.0938 |
| bfs_only | 0.2174 | 0.1328 |
| bfs_gnn | 0.2481 | 0.1484 |

## Interpretation and safeguards

- All rankings and alpha selection obey the original v6 method.
- This is a transductive held-out link task, not clinical validation.
- Sampling from full graph precedes train/test split.
- Only validation was used to select alpha; no test tuning.
- No artificial cluster-label feature added.
- Reduced Python overhead for 100k triples; semantically equivalent to v6.
