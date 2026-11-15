# BFS vs GNN vs validation-calibrated BFS+GNN

- Held-out test queries: 128
- Validation queries: 128
- Candidate counts (min/median/max): {'min': 1, 'median': 105.5, 'max': 270}
- Fusion alpha (GNN weight): **0.70**
- Calibration objective: validation_mrr@10
- BFS fallback selected: False

## TEST (never used to tune alpha)

| Method | K | Hits@K | Recall@K | MRR@K |
|---|---:|---:|---:|---:|
| gnn_only | 1 | 0.1562 | 0.1228 | 0.1562 |
| gnn_only | 3 | 0.3750 | 0.3073 | 0.2500 |
| gnn_only | 5 | 0.4844 | 0.4020 | 0.2746 |
| gnn_only | 10 | 0.5938 | 0.4978 | 0.2906 |
| bfs_only | 1 | 0.1328 | 0.1164 | 0.1328 |
| bfs_only | 3 | 0.3125 | 0.2850 | 0.2096 |
| bfs_only | 5 | 0.3906 | 0.3484 | 0.2280 |
| bfs_only | 10 | 0.5000 | 0.4529 | 0.2416 |
| bfs_gnn | 1 | 0.1797 | 0.1569 | 0.1797 |
| bfs_gnn | 3 | 0.3906 | 0.3506 | 0.2708 |
| bfs_gnn | 5 | 0.4688 | 0.4067 | 0.2880 |
| bfs_gnn | 10 | 0.6016 | 0.5225 | 0.3067 |

## VALIDATION (used for weight selection)

| Method | MRR@10 | Hits@1 |
|---|---:|---:|
| gnn_only | 0.2370 | 0.1250 |
| bfs_only | 0.1868 | 0.0859 |
| bfs_gnn | 0.2673 | 0.1562 |

## Interpretation and safeguards

- All rankings and alpha selection obey the original v6 method.
- This is a transductive held-out link task, not clinical validation.
- Sampling from full graph precedes train/test split.
- Only validation was used to select alpha; no test tuning.
- No artificial cluster-label feature added.
- Reduced Python overhead for 100k triples; semantically equivalent to v6.
