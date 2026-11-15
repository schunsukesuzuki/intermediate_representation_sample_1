# BFS vs GNN vs validation-calibrated BFS+GNN

- Held-out test queries: 128
- Validation queries: 128
- Candidate counts (min/median/max): {'min': 7, 'median': 1008.5, 'max': 1377}
- Fusion alpha (GNN weight): **0.20**
- Calibration objective: validation_mrr@10
- BFS fallback selected: False

## TEST (never used to tune alpha)

| Method | K | Hits@K | Recall@K | MRR@K |
|---|---:|---:|---:|---:|
| gnn_only | 1 | 0.0625 | 0.0521 | 0.0625 |
| gnn_only | 3 | 0.1562 | 0.1149 | 0.1016 |
| gnn_only | 5 | 0.1875 | 0.1332 | 0.1090 |
| gnn_only | 10 | 0.2891 | 0.2041 | 0.1223 |
| bfs_only | 1 | 0.0391 | 0.0247 | 0.0391 |
| bfs_only | 3 | 0.1250 | 0.0849 | 0.0768 |
| bfs_only | 5 | 0.2031 | 0.1580 | 0.0932 |
| bfs_only | 10 | 0.3438 | 0.2852 | 0.1112 |
| bfs_gnn | 1 | 0.0781 | 0.0586 | 0.0781 |
| bfs_gnn | 3 | 0.1641 | 0.1093 | 0.1133 |
| bfs_gnn | 5 | 0.2266 | 0.1722 | 0.1281 |
| bfs_gnn | 10 | 0.3516 | 0.2675 | 0.1448 |

## VALIDATION (used for weight selection)

| Method | MRR@10 | Hits@1 |
|---|---:|---:|
| gnn_only | 0.1181 | 0.0781 |
| bfs_only | 0.1114 | 0.0312 |
| bfs_gnn | 0.1774 | 0.1094 |

## Interpretation and safeguards

- All rankings and alpha selection obey the original v6 method.
- This is a transductive held-out link task, not clinical validation.
- Sampling from full graph precedes train/test split.
- Only validation was used to select alpha; no test tuning.
- No artificial cluster-label feature added.
- Reduced Python overhead for 100k triples; semantically equivalent to v6.
