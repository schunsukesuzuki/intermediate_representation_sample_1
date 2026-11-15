# Offline comparison (SYNTHETIC DATA; NOT Hetionet)

- Nodes: 108; triples: 468
- Seeds: [41, 42, 43]; epochs: 25

| Method | Mean Hits@1 | Mean Hits@10 | Mean MRR@10 | Mean Recall@10 |
|---|---:|---:|---:|---:|
| gnn_only | 0.1607 | 0.9826 | 0.3847 | 0.9714 |
| bfs_only | 0.0308 | 0.4427 | 0.1262 | 0.4034 |
| bfs_gnn | 0.1607 | 0.9826 | 0.3847 | 0.9714 |

## Alpha and per-seed test metrics

| Seed | GNN weight | GNN MRR@10 | BFS MRR@10 | Hybrid MRR@10 |
|---|---:|---:|---:|---:|
| 41 | 0.95 | 0.3473 | 0.1000 | 0.3473 |
| 42 | 0.95 | 0.4080 | 0.1310 | 0.4080 |
| 43 | 0.95 | 0.3986 | 0.1475 | 0.3986 |

### Limitations

- Toy graph deliberately constructed from synthetic clusters; performance is NOT evidence for Hetionet.
- Each seed uses an independent train/validation/test split and a validation-chosen global weight.
- Exact GNN ranking (not HNSW); generated metrics do not measure generation quality.
