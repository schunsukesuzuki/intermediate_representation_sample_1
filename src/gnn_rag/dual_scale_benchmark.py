"""One-command, two-scale Hetionet benchmark and Markdown/JSON/CSV reports.

GNN: GraphSAGE + relation-conditioned exact ranking.
Baselines: train-graph BFS; validation-calibrated percentile-rank fusion.
No Ollama or Voyager required for this exact-ranking benchmark.  The sample
application's separate API continues to support Voyager and local Qwen3.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .compose_prepare import cached_scaled_hetionet_is_usable
from .dataset_archive import resolve_hetionet_source
from .fast_evaluation import compare_three_fast
from .fusion_evaluation import markdown_comparison
from .hetionet_scale import convert_full_hetionet

METHODS = ('gnn_only', 'bfs_only', 'bfs_gnn')
METRICS = ('hits@1', 'hits@3', 'hits@5', 'hits@10', 'recall@10', 'mrr@10')


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def _store_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def prepare_samples(data_root: Path, sizes: tuple[int, ...], *, seed: int,
                    max_nodes: int, focus: str, allow_download: bool) -> tuple[dict[int, Path], dict]:
    sample_dirs = {size: data_root / 'samples' / f'{size}' for size in sizes}
    status = {}
    pending = []
    for size, directory in sample_dirs.items():
        ok, reason = cached_scaled_hetionet_is_usable(
            directory, target=size, max_nodes=max_nodes, seed=seed, focus=focus)
        status[size] = reason
        if not ok:
            pending.append(size)
    source = 'verified_prepared_csv'
    if pending:
        nodes, edges, source = resolve_hetionet_source(data_root, download=allow_download)
        for size in pending:
            print(f'[prepare] sampling {size:,} triples from full Hetionet', flush=True)
            convert_full_hetionet(nodes, edges, sample_dirs[size], target_triples=size,
                                  max_nodes=max_nodes, seed=seed, focus=focus)
            ok, reason = cached_scaled_hetionet_is_usable(
                sample_dirs[size], target=size, max_nodes=max_nodes, seed=seed, focus=focus)
            if not ok:
                raise RuntimeError(f'Sample {size} failed validation: {reason}')
            status[size] = reason
    meta = {'source_mode': source, 'sample_seed': seed,
            'sizes': {str(size): {'sample_dir': str(sample_dirs[size]),
                                   'metadata': json.loads((sample_dirs[size] / 'dataset_meta.json').read_text()),
                                   'nodes_sha256': _sha256_file(sample_dirs[size] / 'nodes.csv'),
                                   'edges_sha256': _sha256_file(sample_dirs[size] / 'edges.csv'),
                                   'status': status[size]}
                      for size in sizes}}
    return sample_dirs, meta


def run_one(dataset: Path, destination: Path, *, seed: int, epochs: int,
            max_positives_per_epoch: int, eval_max_queries: int,
            feature_mode: str) -> dict:
    """Train/score using production v6 model, Polars, and optimized exact ranking."""
    from .data import load_graph  # Polars imported only during actual training
    from .triple_learning import triple_split, train_triples
    from .typed_search import RelationRanker

    destination.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    graph = load_graph(dataset / 'nodes.csv', dataset / 'edges.csv')
    split = triple_split(graph, seed=seed)
    # One undirected node pair must occur in only one split, regardless of relation.
    def pair_keys(triples):
        return {(min(int(h), int(t)), max(int(h), int(t))) for h, _, t in triples}
    train_pairs, valid_pairs, test_pairs = map(
        pair_keys, (split.train, split.validation, split.test))
    if train_pairs & valid_pairs or train_pairs & test_pairs or valid_pairs & test_pairs:
        raise AssertionError('Train/validation/test contain overlapping unordered node pairs')
    print(f'[train] {len(graph.ids):,} nodes / {len(graph.senders):,} triples / '
          f'seed={seed} / {epochs} epochs', flush=True)
    trained = train_triples(graph, split, seed=seed, epochs=epochs,
                           max_positives_per_epoch=max_positives_per_epoch,
                           feature_mode=feature_mode)
    relation_vectors = np.asarray(trained.params['relation_vectors'], dtype=np.float32)
    ranker = RelationRanker(graph.ids, graph.node_kinds, graph.relation_names,
                            trained.embeddings, relation_vectors)
    report, rows = compare_three_fast(graph, split, ranker, seed=seed,
                                      max_queries=eval_max_queries)
    report.update({
        'dataset': 'Hetionet full-source structural sample',
        'input_triples': int(len(graph.senders)), 'node_count': graph.n_nodes,
        'relation_count': len(graph.relation_names), 'feature_mode': feature_mode,
        'seed': seed, 'epochs': epochs, 'training_metrics': trained.metrics,
        'runtime_seconds': round(time.monotonic() - t0, 2),
        'sample_nodes_sha256': _sha256_file(dataset / 'nodes.csv'),
        'sample_edges_sha256': _sha256_file(dataset / 'edges.csv'),
        'train_max_positives_per_epoch': max_positives_per_epoch,
        'evaluation_notes': [
            'Node names and kinds are unmodified original Hetionet attributes.',
            'Train-graph only embeddings and BFS; pair-disjoint held-out triples.',
            'Validation MRR@10 chooses alpha; held-out test not used for tuning.',
            'All candidate tails of relation-compatible type, exact GNN search (not Voyager).',
            'Full source sampled structurally BEFORE split. Different sizes change query/candidate distributions.',
            'Unobserved biological edges can be unlabeled positives; results are not clinical truth.',
            'Positive ranking batch capped per epoch; adjacency uses all Train triples.',
        ],
    })
    _store_json(destination / 'evaluation.json', report)
    (destination / 'evaluation.md').write_text(markdown_comparison(report), encoding='utf-8')
    if not rows:
        raise RuntimeError('No evaluation query records')
    with (destination / 'evaluation_per_query.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(destination / 'gnn_embeddings.npz', ids=np.array(graph.ids),
                        embeddings=trained.embeddings,
                        relation_names=np.array(graph.relation_names),
                        relation_vectors=relation_vectors)
    summary = ', '.join(f"{method}: MRR@10={report['methods'][method]['test']['mrr@10']:.4f}"
                        for method in METHODS)
    print(f'[result] {len(graph.senders):,} triples, seed={seed}: {summary}', flush=True)
    return report


def summarize_runs(reports: list[dict], *, size: int) -> dict:
    if not reports:
        raise ValueError('Cannot aggregate empty benchmark')
    result = {'size': size, 'seeds': [r['seed'] for r in reports],
              'num_seeds': len(reports),
              'mean_candidate_count': float(statistics.mean(
                  r['candidate_counts_test']['median'] for r in reports)),
              'mean_node_count': float(statistics.mean(r['node_count'] for r in reports)),
              'methods': {}}
    for method in METHODS:
        result['methods'][method] = {}
        for metric in METRICS:
            values = [float(r['methods'][method]['test'][metric]) for r in reports]
            result['methods'][method][metric] = {
                'mean': statistics.mean(values),
                'sd': statistics.stdev(values) if len(values) > 1 else None,
                'per_seed': values,
            }
    training_fields = {
        'final_train_loss_mean': [float(r['training_metrics']['train_loss']) for r in reports
                                  if r.get('training_metrics', {}).get('train_loss') is not None],
        'train_triples_mean': [float(r['training_metrics']['train_triples']) for r in reports
                               if r.get('training_metrics', {}).get('train_triples') is not None],
        'runtime_seconds_mean': [float(r['runtime_seconds']) for r in reports
                                 if r.get('runtime_seconds') is not None],
    }
    result['training'] = {key: float(statistics.mean(values))
                          for key, values in training_fields.items() if values}
    result['gnn_minus_bfs_mrr10'] = [
        float(r['methods']['gnn_only']['test']['mrr@10'] - r['methods']['bfs_only']['test']['mrr@10'])
        for r in reports]
    result['fusion_minus_bfs_mrr10'] = [
        float(r['methods']['bfs_gnn']['test']['mrr@10'] - r['methods']['bfs_only']['test']['mrr@10'])
        for r in reports]
    result['alphas'] = [r['calibration']['alpha'] for r in reports]
    return result


def write_overview(output_root: Path, results: dict[str, dict], params: dict) -> None:
    _store_json(output_root / 'overview.json', {'parameters': params, 'sizes': results})
    lines = ['# Hetionet 10k / 100k Triple benchmark', '',
             'GNN（GraphSAGE関係条件付き全件探索）、BFS、Validation選択のBFS＋GNNを比較。', '',
             f"- Sampling seed: {params['sample_seed']}; training seeds: {', '.join(map(str, params['seeds']))}",
             f"- Epochs: {params['epochs']}; positive triples/epoch cap: {params['max_positives_per_epoch']}",
             f"- Evaluation queries per split (max): {params['eval_max_queries']}",
             '- ノード名：Hetionet本来の名称を使用。人工的クラスター特徴なし。',
             '- GNNの厳密ランキングとBFSは同じ候補集合を使用（Voyager近似はこの比較に含まれない）。',
             '', '| Triple数 | ノード数 | 手法 | Hits@10（平均±SD） | MRR@10（平均±SD） |',
             '|---:|---:|---|---:|---:|']
    def fmt(v):
        return f"{v['mean']:.4f}" + (f" ± {v['sd']:.4f}" if v['sd'] is not None else ' (1 seed)')
    for size, summary in sorted(results.items(), key=lambda x: int(x[0])):
        for method in METHODS:
            d = summary['methods'][method]
            lines.append(f"| {int(size):,} | {summary['mean_node_count']:.0f} | {method} | "
                         f"{fmt(d['hits@10'])} | {fmt(d['mrr@10'])} |")
    lines += ['', '## 解釈上の制限', '',
              '規模ごとのサンプルは別途抽出しており、候補ノード数・クエリ分布も変化します。',
              '**MRR絶対値の規模間差を、学習量の因果効果として解釈しないでください。**',
              '同規模内の3方式は同一seed・同一候補集合・同一保留Tripleで比較しています。',
              '融合係数はValidationのみで学習され、Testでの最適化はしていません。',
              '本評価は医学的回答の正確性や臨床的妥当性を検証しません。',
              '', '詳細は `10000/seed42/evaluation.md` など、各サイズ／seedのレポートを参照してください。', '']
    (output_root / 'overview.md').write_text('\n'.join(lines), encoding='utf-8')
    with (output_root / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['triples','nodes','method','hits@10_mean',
                            'hits@10_sd','mrr@10_mean','mrr@10_sd','seeds'])
        writer.writeheader()
        for size, result in sorted(results.items(), key=lambda x: int(x[0])):
            for method, values in result['methods'].items():
                writer.writerow({'triples': size, 'nodes': result['mean_node_count'], 'method': method,
                                 'hits@10_mean': values['hits@10']['mean'],
                                 'hits@10_sd': values['hits@10']['sd'],
                                 'mrr@10_mean': values['mrr@10']['mean'],
                                 'mrr@10_sd': values['mrr@10']['sd'],
                                 'seeds': ','.join(map(str, result['seeds']))})


def benchmark(args) -> dict:
    if len(set(args.sizes)) != len(args.sizes) or any(s < 100 for s in args.sizes):
        raise ValueError('sizes must be distinct and >=100')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError('seeds must be distinct and nonnegative')
    if args.epochs <= 0 or args.eval_max_queries <= 0 or args.train_max_positives <= 0:
        raise ValueError('epochs and evaluation/training caps must be positive')
    args.output_root.mkdir(parents=True, exist_ok=True)
    prepared, sources = prepare_samples(args.data_root, tuple(args.sizes),
                                        seed=args.sample_seed, max_nodes=args.max_nodes,
                                        focus=args.focus, allow_download=not args.no_download)
    params = {'sizes': args.sizes, 'seeds': args.seeds, 'sample_seed': args.sample_seed,
              'max_nodes': args.max_nodes, 'epochs': args.epochs,
              'eval_max_queries': args.eval_max_queries,
              'max_positives_per_epoch': args.train_max_positives,
              'feature_mode': args.feature_mode, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': sources}
    _store_json(args.output_root / 'run_config.json', params)
    # Mark a new run as in progress: do not present an old PDF as its result.
    from .pdf_report import REPORT_NAME
    for old in (args.output_root / REPORT_NAME, args.pdf_output):
        old.unlink(missing_ok=True)
    overall = {}
    for size in args.sizes:
        reports = []
        for seed in args.seeds:
            location = args.output_root / str(size) / f'seed{seed}'
            report = run_one(prepared[size], location, seed=seed, epochs=args.epochs,
                             max_positives_per_epoch=args.train_max_positives,
                             eval_max_queries=args.eval_max_queries,
                             feature_mode=args.feature_mode)
            reports.append(report)
        summary = summarize_runs(reports, size=size)
        _store_json(args.output_root / str(size) / 'summary.json', summary)
        overall[str(size)] = summary
        # preserve completed report after each size even if next run fails
        write_overview(args.output_root, overall, params)
    # PDF contains only scores from this completed run; never old reference results.
    from .pdf_report import REPORT_NAME, generate_pdf_report
    pdf_in_outputs = args.output_root / REPORT_NAME
    generate_pdf_report(pdf_in_outputs, {'parameters': params, 'sizes': overall})
    # The Docker study service mounts the host project root at /app/project_root.
    if pdf_in_outputs.resolve() != args.pdf_output.resolve():
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        copy_tmp = args.pdf_output.with_suffix('.pdf.tmp')
        shutil.copyfile(pdf_in_outputs, copy_tmp)
        copy_tmp.replace(args.pdf_output)
    print(f'[done] report: {args.output_root / "overview.md"}', flush=True)
    print(f'[done] PDF: {args.pdf_output}', flush=True)
    return overall


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Two-scale Hetionet GNN/BFS study')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs/benchmark'))
    parser.add_argument('--sizes', type=int, nargs='+', default=[10000, 100000])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44,45,46])
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--max-nodes', type=int, default=5000)
    parser.add_argument('--focus', default="Alzheimer's disease")
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--eval-max-queries', type=int, default=128)
    parser.add_argument('--train-max-positives', type=int, default=4096)
    parser.add_argument('--feature-mode', choices=['kind_name','kind_only'], default='kind_name')
    parser.add_argument('--pdf-output', type=Path,
                        default=(Path('/app/project_root/Hetionet_10k_100k_comparison.pdf')
                                 if Path('/app/project_root').is_dir()
                                 else Path('Hetionet_10k_100k_comparison.pdf')),
                        help='PDF at the project root (study Docker service supplies the mounted host path)')
    parser.add_argument('--no-download', action='store_true')
    return parser.parse_args(argv)


def main() -> None:
    benchmark(parse_args())


if __name__ == '__main__':
    main()
