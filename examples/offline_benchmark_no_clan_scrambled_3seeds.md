# Hidden-cluster offline comparison (SYNTHETIC DATA; NOT Hetionet)

- Nodes: 108; triples: 468
- Feature mode: kind_only (no cluster or name identity features)
- Node-index scrambling: True
- Seeds: [41, 42, 43]; epochs: 25

| Method | Mean Hits@1 | Mean Hits@10 | Mean MRR@10 | Mean Recall@10 |
|---|---:|---:|---:|---:|
| gnn_only | 0.0211 | 0.1578 | 0.0532 | 0.1276 |
| bfs_only | 0.0259 | 0.5562 | 0.1487 | 0.5025 |
| bfs_gnn | 0.0389 | 0.5216 | 0.1435 | 0.4766 |

## Alpha and per-seed test metrics

| Seed | GNN weight | GNN MRR@10 | BFS MRR@10 | Hybrid MRR@10 |
|---|---:|---:|---:|---:|
| 41 | 0.30 | 0.0548 | 0.1683 | 0.1626 |
| 42 | 0.00 | 0.0465 | 0.1290 | 0.1290 |
| 43 | 0.15 | 0.0584 | 0.1489 | 0.1390 |

### Limitations

- Toy graph topology is intentionally clustered, but latent cluster labels are NEVER model features. Not Hetionet.
- Each seed uses an independent train/validation/test split and a validation-chosen global weight.
- Exact GNN ranking (not HNSW); generated metrics do not measure generation quality.
