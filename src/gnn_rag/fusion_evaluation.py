"""Leakage-controlled comparison: exact GNN vs train-graph BFS vs their fusion.

- GraphSAGE+relation vectors and all adjacency features come from TRAIN only.
- Fusion alpha is calibrated with VALIDATION queries only, before TEST scoring.
- All methods rank *identical* relation-compatible eligible tails per query.
- Rank-percentile fusion keeps dissimilar score scales comparable:
    F_alpha(v) = alpha * P_GNN(v) + (1-alpha) * P_BFS(v)
  with percentiles 1 for rank 1, 0 for last; ties broken stably by BFS,
  then GNN, then node ID. Alpha=0 reproduces BFS; alpha=1 the GNN.
- No ANN truncation in this correctness benchmark. Voyager is still available
  in the live evidence pipeline, but ANN and learned ranking must be separated.
"""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ontology import RELATIONS
from .triple_evaluation import _bfs_distances, _bfs_rank, _measures
from .triple_learning import candidate_without_observed


@dataclass(frozen=True)
class QueryRanks:
    head: int
    relation: int
    candidates: tuple[int, ...]
    positives: frozenset[int]
    gnn: tuple[int, ...]
    bfs: tuple[int, ...]


def _target_groups(graph: Any, triples: np.ndarray) -> dict[tuple[int, int], set[int]]:
    groups: dict[tuple[int, int], set[int]] = defaultdict(set)
    for head, rel, tail in np.asarray(triples).reshape(-1, 3):
        h, r, t = int(head), int(rel), int(tail)
        groups[h, r].add(t)
        spec = RELATIONS.get(graph.relation_names[r])
        if spec is not None and not spec.directed and spec.source_kind == spec.target_kind:
            groups[t, r].add(h)
    return groups


def make_query_ranks(graph: Any, split: Any, ranker: Any, *, partition: str,
                     max_queries: int | None = None, seed: int = 42) -> list[QueryRanks]:
    """Construct equal candidate sets using TRAIN graph for BOTH ranking methods.

    Validation labels may only be filtered when evaluating TEST. The test labels
    are never consulted in model fitting, calibration, or graph construction.
    """
    if partition not in ('validation', 'test'):
        raise ValueError('partition must be validation or test')
    labels = _target_groups(graph, getattr(split, partition))
    validation = _target_groups(graph, split.validation) if partition == 'test' else {}
    adjacency = _bfs_distances(split.train, graph.n_nodes)
    # Preselect validation/test queries by relation-stratified, deterministic
    # hashing. This uses query IDs and relation only, never their rank/labels.
    keys = sorted(labels)
    if max_queries is not None:
        if max_queries < 1:
            raise ValueError('max_queries must be >=1')
        by_relation = defaultdict(list)
        for h, r in keys:
            digest = hashlib.sha256(f'{seed}|{partition}|{graph.ids[h]}|{graph.relation_names[r]}'.encode()).digest()
            by_relation[r].append((digest, (h, r)))
        for r in by_relation:
            by_relation[r].sort()
        chosen = []
        rels = sorted(by_relation)
        while len(chosen) < max_queries and rels:
            active = []
            for r in rels:
                if by_relation[r] and len(chosen) < max_queries:
                    chosen.append(by_relation[r].pop(0)[1])
                if by_relation[r]:
                    active.append(r)
            rels = active
        keys = chosen
    distance_cache = {}
    output = []
    for (head, relation) in keys:
        relevant = labels[(head, relation)]
        eligible = candidate_without_observed(graph, head, relation, split.train)
        # Standard filtered ranking: other known positives in VALIDATION are
        # not counted as negative on TEST, but are NOT used for tuning alpha.
        suppressed = validation.get((head, relation), set())
        candidates = np.asarray([int(i) for i in eligible if int(i) not in suppressed], dtype=np.int32)
        if not relevant.issubset(set(candidates.tolist())):
            raise ValueError('Held-out relevant tail is absent from candidate set')
        if not len(candidates):
            continue
        expected = set(candidates.tolist())
        ranked_gnn = tuple(int(i) for i in ranker.exact(head, relation, candidates))
        if head not in distance_cache:
            distances = {head: 0}
            queue = deque([head])
            while queue:
                u = queue.popleft()
                for v in adjacency[u]:
                    if v not in distances:
                        distances[v] = distances[u] + 1
                        queue.append(v)
            distance_cache[head] = distances
        d = distance_cache[head]
        ranked_bfs = tuple(sorted((int(v) for v in candidates),
                           key=lambda v: (d.get(v, float('inf')), -len(adjacency[v]), v)))
        if len(ranked_gnn) != len(expected) or set(ranked_gnn) != expected:
            raise ValueError('GNN full ranking must be a permutation of all candidates')
        if len(ranked_bfs) != len(expected) or set(ranked_bfs) != expected:
            raise ValueError('BFS full ranking must be a permutation of all candidates')
        output.append(QueryRanks(head, relation, tuple(map(int, candidates)),
                                 frozenset(relevant), ranked_gnn, ranked_bfs))
    return output


def fused_ranking(query: QueryRanks, alpha: float) -> tuple[int, ...]:
    """Fuse percentile ranks; alpha learned only on validation queries."""
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    n = len(query.candidates)
    g_position = {v: p for p, v in enumerate(query.gnn)}
    b_position = {v: p for p, v in enumerate(query.bfs)}
    if alpha == 0.:
        return query.bfs
    if alpha == 1.:
        return query.gnn
    scale = max(n - 1, 1)
    def key(node: int):
        g_rank, b_rank = g_position[node], b_position[node]
        score = alpha * (1. - g_rank / scale) + (1. - alpha) * (1. - b_rank / scale)
        return (-score, b_rank, g_rank, node)
    return tuple(sorted(query.candidates, key=key))


def _aggregate(queries: list[QueryRanks], alpha: float, ks: tuple[int, ...]) -> dict[str, float]:
    if not queries:
        raise ValueError('Cannot aggregate an empty query set')
    metrics = {name: [] for k in ks for name in (f'hits@{k}', f'recall@{k}', f'mrr@{k}')}
    for q in queries:
        for name, value in _measures(list(fused_ranking(q, alpha)), set(q.positives), ks).items():
            metrics[name].append(value)
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def calibrate_alpha(validation: list[QueryRanks], *, objective_k: int = 10,
                    grid: tuple[float, ...] | None = None) -> dict[str, Any]:
    """Tune a SINGLE global mixture weight; no test labels or candidate inputs.

    If alpha=0 wins, the learned combination is correctly reported as a BFS
    fallback, rather than inventing a gain from GNN.
    """
    if not validation:
        raise ValueError('Cannot tune fusion without validation queries')
    if objective_k < 1:
        raise ValueError('objective_k must be positive')
    weights = grid if grid is not None else tuple(i / 20 for i in range(21))
    if not weights or any(not 0 <= a <= 1 for a in weights):
        raise ValueError('Invalid alpha grid')
    results = []
    for alpha in sorted(set(weights)):
        m = _aggregate(validation, float(alpha), (1, objective_k))
        results.append({'alpha': float(alpha), 'validation_mrr': m[f'mrr@{objective_k}'],
                        'validation_hits1': m['hits@1']})
    # Optimize macro-MRR; a tie prefers a lower alpha to limit complexity.
    best = max(results, key=lambda x: (x['validation_mrr'], x['validation_hits1'], -x['alpha']))
    return {'alpha': best['alpha'], 'objective': f'validation_mrr@{objective_k}',
            'grid': results, 'validation_queries': len(validation),
            'uses_test_labels': False,
            'fallback_to_bfs': best['alpha'] == 0.,
            'chosen_validation_mrr': best['validation_mrr']}


def compare_three(graph: Any, split: Any, ranker: Any,
                  *, ks: tuple[int, ...] = (1, 3, 5, 10),
                  tune_k: int = 10, seed: int = 42,
                  max_queries: int | None = None) -> tuple[dict, list[dict]]:
    """Validation selects alpha, untouched test compares exactly three methods."""
    if not ks or min(ks) < 1:
        raise ValueError('ks must contain positive integers')
    ks = tuple(sorted(set(ks)))
    validation = make_query_ranks(graph, split, ranker, partition='validation',
                                  max_queries=max_queries, seed=seed)
    if not validation:
        raise ValueError('No validation queries; increase graph size')
    calibration = calibrate_alpha(validation, objective_k=tune_k)
    test = make_query_ranks(graph, split, ranker, partition='test',
                             max_queries=max_queries, seed=seed)
    if not test:
        raise ValueError('No test queries; increase graph size')
    alpha = float(calibration['alpha'])
    methods = {'gnn_only': 1., 'bfs_only': 0., 'bfs_gnn': alpha}
    metrics = {name: _aggregate(test, a, ks) for name, a in methods.items()}
    rows = []
    for q in test:
        for name, a in methods.items():
            scores = _measures(list(fused_ranking(q, a)), set(q.positives), ks)
            rows.append({'method': name, 'source': graph.ids[q.head],
                         'relation': graph.relation_names[q.relation],
                         'n_candidates': len(q.candidates),
                         'n_relevant': len(q.positives),
                         'alpha': a, **scores})
    val_ref = {'gnn_only': _aggregate(validation, 1., ks),
               'bfs_only': _aggregate(validation, 0., ks),
               'bfs_gnn': _aggregate(validation, alpha, ks)}
    report = {'task': 'held_out_relation_typed_triple_retrieval',
              'dataset': 'set by calling pipeline', 'model': 'GraphSAGE + relation query',
              'comparison': 'GNN only vs BFS only vs validation-tuned BFS+GNN',
              'methods': {name: {'test': metrics[name], 'validation': val_ref[name]}
                          for name in methods},
              'calibration': calibration, 'query_count': len(test),
              'evaluation_query_cap_per_split': max_queries,
              'validation_query_sampled': max_queries is not None,
              'test_query_sampled': max_queries is not None,
              'validation_query_count': len(validation),
              'candidate_counts_test': {'min': min(len(q.candidates) for q in test),
                                        'median': float(np.median([len(q.candidates) for q in test])),
                                        'max': max(len(q.candidates) for q in test)},
              'test_triples': len(split.test), 'validation_triples': len(split.validation),
              'train_triples': len(split.train), 'ks': list(ks), 'seed': seed,
              'fusion_formula': 'alpha*GNN_rank_percentile + (1-alpha)*BFS_rank_percentile',
              'candidate_rule': 'same relation-compatible tails excluding train links; validation positives also filtered on test',
              'bfs_graph': 'train triples only, undirected adjacency; distance then degree then ID',
              'gnn_retrieval': 'exact all-candidate relation-conditioned score (NOT HNSW)',
              'notes': [
                  'No test labels used to tune alpha; one global weight calibrated on validation MRR.',
                  'If calibrated alpha=0, bfs_gnn is BFS fallback, not evidence of improvement.',
                  'All three methods use exactly the same per-query candidate list and relevant tails.',
                  'Transductive held-out link recovery, not biomedical truth or LLM answer quality.',
                  'Unknown relevant links can be mislabeled as negatives in incomplete KGs.',
                  'One split/seed is descriptive only; replicate across seeds for robust comparison.',
                  'BFS and GNN features see training triples only; node names/kinds are public attributes.',
              ]}
    return report, rows


def markdown_comparison(report: dict) -> str:
    c = report['calibration']
    lines = ['# BFS vs GNN vs validation-calibrated BFS+GNN', '',
             f"- Held-out test queries: {report['query_count']}",
             f"- Validation queries: {report['validation_query_count']}",
             f"- Candidate counts (min/median/max): {report['candidate_counts_test']}",
             f"- Fusion alpha (GNN weight): **{c['alpha']:.2f}**",
             f"- Calibration objective: {c['objective']}",
             f"- BFS fallback selected: {c['fallback_to_bfs']}",
             '', '## TEST (never used to tune alpha)', '',
             '| Method | K | Hits@K | Recall@K | MRR@K |',
             '|---|---:|---:|---:|---:|']
    for name, result in report['methods'].items():
        for k in report['ks']:
            m = result['test']
            lines.append(f"| {name} | {k} | {m[f'hits@{k}']:.4f} | {m[f'recall@{k}']:.4f} | {m[f'mrr@{k}']:.4f} |")
    lines += ['', '## VALIDATION (used for weight selection)', '',
              '| Method | MRR@10 | Hits@1 |', '|---|---:|---:|']
    for name, result in report['methods'].items():
        m = result['validation']
        lines.append(f"| {name} | {m.get('mrr@10', float('nan')):.4f} | {m['hits@1']:.4f} |")
    lines += ['', '## Interpretation and safeguards', '']
    lines += ['- ' + note for note in report['notes']]
    return '\n'.join(lines) + '\n'


def rank_candidates(ranker: Any, train_triples: np.ndarray, head: int, relation: int,
                    candidates: np.ndarray, *, alpha: float, method: str = 'bfs_gnn') -> list[int]:
    """Use the same benchmark ranking rules at inference, without held-out labels."""
    if method not in ('gnn_only', 'bfs_only', 'bfs_gnn'):
        raise ValueError('Unknown rank method')
    allowed = np.asarray(candidates, dtype=np.int32)
    if not len(allowed):
        return []
    if len(set(allowed.tolist())) != len(allowed):
        raise ValueError('Duplicate candidate IDs')
    gnn = tuple(int(i) for i in ranker.exact(head, relation, allowed))
    bfs = tuple(_bfs_rank(head, allowed, _bfs_distances(train_triples, len(ranker.ids))))
    query = QueryRanks(head, relation, tuple(allowed.tolist()), frozenset(), gnn, bfs)
    return list(fused_ranking(query, {'gnn_only': 1., 'bfs_only': 0., 'bfs_gnn': alpha}[method]))


# Large-graph compatible evaluator. This version is numerically verified to
# match the original implementation on Hetionet 10k for five seeds (42..46).
# Keep the reference implementation available for test/equivalence analysis.
compare_three_reference = compare_three

def compare_three(graph, split, ranker, *, ks=(1, 3, 5, 10), tune_k=10,
                  seed=42, max_queries=None):
    from .fast_evaluation import compare_three_fast
    return compare_three_fast(graph, split, ranker, ks=ks, tune_k=tune_k,
                              seed=seed, max_queries=max_queries)
