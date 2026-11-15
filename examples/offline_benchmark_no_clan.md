# Hidden-cluster offline comparison (SYNTHETIC DATA; NOT Hetionet)

- Nodes: 108; triples: 468
- Feature mode: kind_only (no cluster or name identity features)
- Seeds: [41, 42, 43]; epochs: 25

| Method | Mean Hits@1 | Mean Hits@10 | Mean MRR@10 | Mean Recall@10 |
|---|---:|---:|---:|---:|
| gnn_only | 0.0359 | 0.3033 | 0.1006 | 0.2491 |
| bfs_only | 0.0308 | 0.4427 | 0.1262 | 0.4034 |
| bfs_gnn | 0.0578 | 0.5081 | 0.1689 | 0.4343 |

## Alpha and per-seed test metrics

| Seed | GNN weight | GNN MRR@10 | BFS MRR@10 | Hybrid MRR@10 |
|---|---:|---:|---:|---:|
| 41 | 0.50 | 0.0946 | 0.1000 | 0.2002 |
| 42 | 0.00 | 0.0611 | 0.1310 | 0.1310 |
| 43 | 0.60 | 0.1460 | 0.1475 | 0.1757 |

### Limitations

- Toy graph topology is intentionally clustered, but latent cluster labels are NEVER model features. Not Hetionet.
- Each seed uses an independent train/validation/test split and a validation-chosen global weight.
- Exact GNN ranking (not HNSW); generated metrics do not measure generation quality.
