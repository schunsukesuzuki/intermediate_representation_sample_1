# Hidden-cluster offline comparison (SYNTHETIC DATA; NOT Hetionet)

- Nodes: 108; triples: 468
- Feature mode: kind_only (no cluster or name identity features)
- Seeds: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]; epochs: 25

| Method | Mean Hits@1 | Mean Hits@10 | Mean MRR@10 | Mean Recall@10 |
|---|---:|---:|---:|---:|
| gnn_only | 0.0325 | 0.2400 | 0.0794 | 0.1956 |
| bfs_only | 0.0378 | 0.5124 | 0.1461 | 0.4526 |
| bfs_gnn | 0.0617 | 0.5367 | 0.1744 | 0.4685 |

## Alpha and per-seed test metrics

| Seed | GNN weight | GNN MRR@10 | BFS MRR@10 | Hybrid MRR@10 |
|---|---:|---:|---:|---:|
| 41 | 0.50 | 0.0946 | 0.1000 | 0.2002 |
| 42 | 0.00 | 0.0611 | 0.1310 | 0.1310 |
| 43 | 0.60 | 0.1460 | 0.1475 | 0.1757 |
| 44 | 0.50 | 0.0813 | 0.1818 | 0.2445 |
| 45 | 0.35 | 0.0943 | 0.1379 | 0.2084 |
| 46 | 0.20 | 0.0517 | 0.1499 | 0.1021 |
| 47 | 0.15 | 0.0654 | 0.1360 | 0.1348 |
| 48 | 0.25 | 0.0866 | 0.1650 | 0.1811 |
| 49 | 0.00 | 0.0396 | 0.1478 | 0.1478 |
| 50 | 0.25 | 0.0737 | 0.1638 | 0.2183 |

### Limitations

- Toy graph topology is intentionally clustered, but latent cluster labels are NEVER model features. Not Hetionet.
- Each seed uses an independent train/validation/test split and a validation-chosen global weight.
- Exact GNN ranking (not HNSW); generated metrics do not measure generation quality.
