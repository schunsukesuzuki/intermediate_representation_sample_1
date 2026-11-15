# BFS vs GNN vs validation-calibrated BFS+GNN

- Held-out test queries: 128
- Validation queries: 128
- Candidate counts (min/median/max): {'min': 2, 'median': 179.0, 'max': 270}
- Fusion alpha (GNN weight): **0.15**
- Calibration objective: validation_mrr@10
- BFS fallback selected: False

## TEST (never used to tune alpha)

| Method | K | Hits@K | Recall@K | MRR@K |
|---|---:|---:|---:|---:|
| gnn_only | 1 | 0.1797 | 0.1274 | 0.1797 |
| gnn_only | 3 | 0.3750 | 0.2937 | 0.2630 |
| gnn_only | 5 | 0.4297 | 0.3428 | 0.2755 |
| gnn_only | 10 | 0.5859 | 0.4757 | 0.2961 |
| bfs_only | 1 | 0.1328 | 0.1103 | 0.1328 |
| bfs_only | 3 | 0.2578 | 0.2319 | 0.1875 |
| bfs_only | 5 | 0.3281 | 0.3066 | 0.2027 |
| bfs_only | 10 | 0.4375 | 0.3960 | 0.2160 |
| bfs_gnn | 1 | 0.1328 | 0.1142 | 0.1328 |
| bfs_gnn | 3 | 0.2734 | 0.2403 | 0.1953 |
| bfs_gnn | 5 | 0.3516 | 0.3159 | 0.2129 |
| bfs_gnn | 10 | 0.4688 | 0.4152 | 0.2269 |

## VALIDATION (used for weight selection)

| Method | MRR@10 | Hits@1 |
|---|---:|---:|
| gnn_only | 0.2431 | 0.1484 |
| bfs_only | 0.2335 | 0.1484 |
| bfs_gnn | 0.2500 | 0.1406 |

## Interpretation and safeguards

- All rankings and alpha selection obey the original v6 method.
- This is a transductive held-out link task, not clinical validation.
- Sampling from full graph precedes train/test split.
- Only validation was used to select alpha; no test tuning.
- No artificial cluster-label feature added.
- Reduced Python overhead for 100k triples; semantically equivalent to v6.
