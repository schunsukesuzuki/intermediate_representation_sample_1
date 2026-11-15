"""Optional multi-seed Hetionet comparison in Docker (not part of regular up)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

METHODS = ('gnn_only', 'bfs_only', 'bfs_gnn')
METRICS = ('hits@1', 'hits@3', 'hits@10', 'mrr@3', 'mrr@10', 'recall@10')


def summarize_reports(items: list[dict]) -> dict:
    """Descriptive mean/std and paired gaps over independent seeds, no p-values."""
    if not items:
        raise ValueError('No per-seed reports')
    if len({item['seed'] for item in items}) != len(items):
        raise ValueError('Repeated seed in experiment')
    aggregate = {'seeds': [int(item['seed']) for item in items], 'runs': len(items),
                 'methods': {}, 'paired_mrr10_delta': {},
                 'scope': 'descriptive macro metrics only; same candidate protocol per seed'}
    for method in METHODS:
        aggregate['methods'][method] = {}
        for metric in METRICS:
            x = np.asarray([item['methods'][method]['test'][metric] for item in items], dtype=float)
            aggregate['methods'][method][metric] = {
                'mean': float(x.mean()),
                'std': float(x.std(ddof=1)) if len(x) > 1 else None}
    for baseline in ('bfs_only', 'gnn_only'):
        x = np.asarray([item['methods']['bfs_gnn']['test']['mrr@10'] -
                        item['methods'][baseline]['test']['mrr@10'] for item in items], dtype=float)
        aggregate['paired_mrr10_delta'][f'bfs_gnn_minus_{baseline}'] = {
            'per_seed': x.tolist(), 'mean': float(x.mean()),
            'positive_seeds': int((x > 0).sum()), 'negative_seeds': int((x < 0).sum())}
    return aggregate


def markdown_multiseed(aggregate: dict) -> str:
    out = ['# Hetionet triple retrieval — multi-seed descriptive benchmark', '',
           f"- Split/training seeds: {aggregate['seeds']}",
           '- Same candidate set for GNN, BFS, hybrid **within each seed**',
           '- Alpha selected using validation per seed; only test scores aggregated',
           '', '| Method | Hits@1 mean±std | Hits@10 mean±std | MRR@10 mean±std |',
           '|---|---:|---:|---:|']
    for name, data in aggregate['methods'].items():
        def fmt(metric):
            entry = data[metric]
            return f"{entry['mean']:.4f} ± {entry['std']:.4f}" if entry['std'] is not None else f"{entry['mean']:.4f}"
        out.append(f"| {name} | {fmt('hits@1')} | {fmt('hits@10')} | {fmt('mrr@10')} |")
    out.extend(['', '## Paired per-seed differences in test MRR@10', ''])
    for label, value in aggregate['paired_mrr10_delta'].items():
        out.append(f"- {label}: mean {value['mean']:.4f}; improves in {value['positive_seeds']} / {aggregate['runs']} seeds; differences {value['per_seed']}")
    out.extend(['', '## Limits', '',
                '- These are independent random splits of the **same selected subgraph**, not independent medical cohorts.',
                '- Small Hetionet focus subgraphs may overrepresent short graph distances and relations.',
                '- Standard deviations are descriptive. Do not interpret as clinical accuracy, statistical significance, or causal validity.',
                '- A validation weight of zero is BFS-only fallback, not genuine fusion improvement.', ''])
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=Path, default=Path('/app/data/hetionet_demo/nodes.csv'))
    parser.add_argument('--edges', type=Path, default=Path('/app/data/hetionet_demo/edges.csv'))
    parser.add_argument('--output', type=Path, default=Path('/app/outputs/multiseed'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[41, 42, 43])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--eval-max-queries', type=int, default=128)
    parser.add_argument('--train-max-positives', type=int, default=4096)
    parser.add_argument('--feature-mode', choices=['kind_only','kind_name'], default='kind_name')
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError('Provide at least 2 distinct seeds')
    if not args.nodes.is_file() or not args.edges.is_file():
        raise FileNotFoundError('Prepared Hetionet CSV missing. First run docker compose up --build (prepare-data).')
    from .rag import run_rag
    reports = []
    for seed in args.seeds:
        target = args.output / f'seed_{seed}'
        result = run_rag(output=target, nodes_csv=args.nodes, edges_csv=args.edges,
                         epochs=args.epochs, seed=seed, generate=False, evaluate=True,
                         eval_max_queries=args.eval_max_queries,
                         train_max_positives=args.train_max_positives,
                         feature_mode=args.feature_mode)
        report = json.loads((target / 'evaluation.json').read_text(encoding='utf-8'))
        report['seed'] = seed
        reports.append(report)
        print(f"seed {seed}, alpha={report['calibration']['alpha']} test_MRR10=" +
              str({method: round(m['test']['mrr@10'], 4) for method, m in report['methods'].items()}), flush=True)
    summary = summarize_reports(reports)
    summary.update({'dataset': 'Hetionet subset from provided CSV paths', 'epochs': args.epochs,
                    'nodes': str(args.nodes), 'edges': str(args.edges)})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (args.output / 'summary.md').write_text(markdown_multiseed(summary), encoding='utf-8')
    print('Wrote', args.output / 'summary.md', flush=True)


if __name__ == '__main__':
    main()
