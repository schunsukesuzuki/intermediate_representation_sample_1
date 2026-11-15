"""Polars/Voyager-free JAX benchmark on an explicitly SYNTHETIC typed KG.

Usage: PYTHONPATH=src python scripts/offline_compare.py --seeds 41 42 43
Creates examples/offline_benchmark.json and .md by default.
Do not present these toy numbers as Hetionet results.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from gnn_rag.triple_learning import triple_split, train_triples
from gnn_rag.typed_search import RelationRanker
from gnn_rag.fusion_evaluation import compare_three, markdown_comparison


class Column:
    def __init__(self, values): self.values = values
    def to_list(self): return list(self.values)


class Nodes:
    def __init__(self, names, topics): self.columns = {'name': names, 'topic': topics}
    def get_column(self, key): return Column(self.columns[key])


@dataclass
class MiniGraph:
    ids: tuple[str, ...]
    node_kinds: tuple[str, ...]
    relation_names: tuple[str, ...]
    senders: np.ndarray
    receivers: np.ndarray
    relation_ids: np.ndarray
    weights: np.ndarray
    nodes: Nodes

    @property
    def n_nodes(self): return len(self.ids)


def make_synthetic_kg(n_disease: int = 36, n_gene: int = 72, *,
                      scramble_indices: bool = False,
                      permutation_seed: int = 20261114) -> MiniGraph:
    """Synthetic typed graph with hidden latent blocks; NOT accessible as features."""
    if n_disease < 12 or n_gene < 24:
        raise ValueError('Too few synthetic nodes')
    n_clans = 6
    ids = tuple(f'd{i:03}' for i in range(n_disease)) + tuple(f'g{i:03}' for i in range(n_gene))
    kinds = ('Disease',) * n_disease + ('Gene',) * n_gene
    # Names deliberately carry no cluster or identity features; the graph's
    # connectivity is generated from hidden groups as in the original toy task.
    names = (('Synthetic disease',) * n_disease + ('Synthetic gene',) * n_gene)
    triples = set()
    # Disease-Gene edges: mostly within clan, plus a few cross-clan links.
    for d in range(n_disease):
        clan = d % n_clans
        same = [g for g in range(n_gene) if g % n_clans == clan]
        for j in range(6):
            triples.add((d, 0, n_disease + same[(d // n_clans * 3 + j * 2) % len(same)]))
        triples.add((d, 0, n_disease + (d * 7 + 3) % n_gene))
        # Disease-Disease similarity within clan plus one across-clan edge.
        triples.add((d, 1, (d + n_clans) % n_disease))
        triples.add((d, 1, (d + 2 * n_clans) % n_disease))
    for g in range(n_gene):
        clan = g % n_clans
        same = [j for j in range(n_gene) if j % n_clans == clan]
        triples.add((n_disease + g, 2, n_disease + same[(g // n_clans + 1) % len(same)]))
        triples.add((n_disease + g, 2, n_disease + same[(g // n_clans + 3) % len(same)]))
    a = np.asarray(sorted(t for t in triples if t[0] != t[2]), dtype=np.int32)
    if scramble_indices:
        # Remap node indices WITHIN each node kind, independently of the
        # latent clan. This removes even the arithmetic i % n_clans pattern
        # from public node IDs and index-based BFS tie breaking.
        rng = np.random.default_rng(permutation_seed)
        mapping = np.concatenate((rng.permutation(n_disease),
                                  n_disease + rng.permutation(n_gene)))
        a = a.copy()
        a[:, 0] = mapping[a[:, 0]]
        a[:, 2] = mapping[a[:, 2]]
        a = np.asarray(sorted(map(tuple, a.tolist())), dtype=np.int32)
    return MiniGraph(ids, kinds, ('DaG', 'DrD', 'GiG'), a[:, 0], a[:, 2], a[:, 1],
                     np.ones(len(a), dtype=np.float32), Nodes(names, kinds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--scramble-indices', action=argparse.BooleanOptionalAction, default=True,
                        help='Scramble node-index mapping by default; --no-scramble-indices reproduces the paired feature ablation')
    parser.add_argument('--permutation-seed', type=int, default=20261114)
    parser.add_argument('--feature-mode', choices=('kind_only', 'kind_name'), default='kind_only',
                        help='kind_only (default): no names or latent group in node features')
    parser.add_argument('--output', type=Path, default=Path('examples/offline_benchmark_strict.json'))
    args = parser.parse_args()
    graph = make_synthetic_kg(scramble_indices=args.scramble_indices,
                              permutation_seed=args.permutation_seed)
    assert len(set(graph.nodes.get_column('name').to_list())) == 2
    records = []
    for seed in args.seeds:
        split = triple_split(graph, seed=seed)
        model = train_triples(graph, split, seed=seed, epochs=args.epochs,
                              feature_mode=args.feature_mode)
        ranker = RelationRanker(graph.ids, graph.node_kinds, graph.relation_names,
                                model.embeddings, np.asarray(model.params['relation_vectors']))
        comparison, _ = compare_three(graph, split, ranker, seed=seed)
        records.append({'seed': seed, 'split': {'train': len(split.train),
                                              'validation': len(split.validation),
                                              'test': len(split.test)},
                        'model_train_loss': model.metrics['train_loss'], **comparison})
        print('seed', seed, 'alpha', comparison['calibration']['alpha'], 'test MRR@10',
              {key: round(val['test']['mrr@10'], 4) for key, val in comparison['methods'].items()}, flush=True)
    result = {'dataset': 'SYNTHETIC typed graph with HIDDEN blocks (NOT Hetionet)',
              'nodes': graph.n_nodes, 'triples': len(graph.senders),
              'model': 'GraphSAGE/relational ranker, same architecture as v5',
              'feature_mode': args.feature_mode,
              'features_do_not_include_latent_cluster': True,
              'node_name_values': sorted(set(graph.nodes.get_column('name').to_list())),
              'scramble_indices': args.scramble_indices,
              'permutation_seed': args.permutation_seed if args.scramble_indices else None,
              'seeds': args.seeds, 'epochs': args.epochs, 'runs': records,
              'limitation': 'Synthetic graph construction still uses hidden blocks; GNN has no cluster tags. No live Voyager, Polars, Hetionet or Docker verification.'}
    result['mean_test_metrics'] = {
        method: {metric: float(np.mean([r['methods'][method]['test'][metric] for r in records]))
                 for metric in ('hits@1', 'hits@3', 'hits@10', 'mrr@3', 'mrr@10', 'recall@10')}
        for method in ('gnn_only', 'bfs_only', 'bfs_gnn')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# Hidden-cluster offline comparison (SYNTHETIC DATA; NOT Hetionet)', '',
             f"- Nodes: {result['nodes']}; triples: {result['triples']}",
             f"- Feature mode: {args.feature_mode} (no cluster or name identity features)",
             f"- Node-index scrambling: {args.scramble_indices}",
             f"- Seeds: {args.seeds}; epochs: {args.epochs}",
             '', '| Method | Mean Hits@1 | Mean Hits@10 | Mean MRR@10 | Mean Recall@10 |',
             '|---|---:|---:|---:|---:|']
    for method, values in result['mean_test_metrics'].items():
        lines.append(f"| {method} | {values['hits@1']:.4f} | {values['hits@10']:.4f} | {values['mrr@10']:.4f} | {values['recall@10']:.4f} |")
    lines.extend(['', '## Alpha and per-seed test metrics', '',
                  '| Seed | GNN weight | GNN MRR@10 | BFS MRR@10 | Hybrid MRR@10 |',
                  '|---|---:|---:|---:|---:|'])
    for r in records:
        m = r['methods']
        lines.append(f"| {r['seed']} | {r['calibration']['alpha']:.2f} | {m['gnn_only']['test']['mrr@10']:.4f} | {m['bfs_only']['test']['mrr@10']:.4f} | {m['bfs_gnn']['test']['mrr@10']:.4f} |")
    lines.extend(['', '### Limitations', '', '- Toy graph topology is intentionally clustered, but latent cluster labels are NEVER model features. Not Hetionet.',
                  '- Each seed uses an independent train/validation/test split and a validation-chosen global weight.',
                  '- Exact GNN ranking (not HNSW); generated metrics do not measure generation quality.'])
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
