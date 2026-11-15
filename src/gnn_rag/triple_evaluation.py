"""Fair relation-conditioned, target-type restricted, held-out triple ranking.

ALL methods are scored over the exact same candidate set and test targets.
Train graph only for model fitting and BFS; validation labels filtered at report
construction only to avoid treating known other positives as irrelevant.
This measures unobserved KG triple retrieval, not LLM answer quality.
"""
from __future__ import annotations

from collections import defaultdict, deque
import numpy as np

from .triple_learning import candidate_without_observed
from .ontology import RELATIONS


def _bfs_distances(train: np.ndarray, n_nodes: int):
    adjacency = [set() for _ in range(n_nodes)]
    for s, _, t in train:
        adjacency[int(s)].add(int(t))
        adjacency[int(t)].add(int(s))
    return adjacency


def _bfs_rank(head: int, choices: np.ndarray, adjacency: list[set[int]]) -> list[int]:
    distance = {int(head): 0}
    queue = deque([int(head)])
    while queue:
        node = queue.popleft()
        for other in sorted(adjacency[node]):
            if other not in distance:
                distance[other] = distance[node] + 1
                queue.append(other)
    return sorted((int(i) for i in choices),
                  key=lambda v: (distance.get(v, float('inf')), -len(adjacency[v]), v))


def _measures(order: list[int], positives: set[int], ks: tuple[int, ...]) -> dict[str, float]:
    result = {}
    for k in ks:
        match = [position+1 for position, node in enumerate(order[:k]) if node in positives]
        result.update({f'hits@{k}': float(bool(match)),
                       f'recall@{k}': len(match)/len(positives),
                       f'mrr@{k}': 1/min(match) if match else 0.})
    return result


def evaluate_triples(graph, split, ranker, *, seed: int = 42,
                     ks: tuple[int, ...] = (1, 3, 5, 10),
                     random_repeats: int = 20) -> tuple[dict, list[dict]]:
    if min(ks) < 1 or random_repeats < 1:
        raise ValueError('Invalid retrieval evaluation settings')
    ks = tuple(sorted(set(ks)))
    targets: dict[tuple[int, int], set[int]] = defaultdict(set)
    for h, r, t in split.test:
        h, r, t = int(h), int(r), int(t)
        targets[h, r].add(t)
        spec = RELATIONS.get(graph.relation_names[r])
        if spec is not None and not spec.directed and spec.source_kind == spec.target_kind:
            targets[t, r].add(h)
    val_targets: dict[tuple[int, int], set[int]] = defaultdict(set)
    for h, r, t in split.validation:
        h, r, t = int(h), int(r), int(t)
        val_targets[h, r].add(t)
        spec = RELATIONS.get(graph.relation_names[r])
        if spec is not None and not spec.directed and spec.source_kind == spec.target_kind:
            val_targets[t, r].add(h)
    adj = _bfs_distances(split.train, graph.n_nodes)
    scores: dict[str, list[dict]] = {name: [] for name in ('gnn_exact', 'gnn_voyager', 'gnn_no_relation', 'graph_bfs', 'random')}
    rows = []
    skipped = 0
    for (head, rel), positives in sorted(targets.items()):
        candidates = candidate_without_observed(graph, head, rel, split.train)
        # Standard filtered evaluation: remove validation positives from the
        # candidate list ONLY at metric-time; never use them as training inputs.
        candidates = np.asarray([i for i in candidates if i not in val_targets[head, rel]], dtype=np.int32)
        if not positives.issubset(set(candidates.tolist())):
            raise ValueError('Withheld target was excluded from the evaluation candidates')
        if not len(candidates):
            skipped += 1
            continue
        k = min(max(ks), len(candidates))
        exact = ranker.exact(head, rel, candidates, k)
        approx = ranker.voyager(head, rel, candidates, k)
        bfs = _bfs_rank(head, candidates, adj)[:k]
        # Query ablation on the very same trained embeddings/candidate sets:
        # disable relation translation at inference (not a separately trained v3 model).
        no_relation = sorted((int(i) for i in candidates),
                             key=lambda v: (-float(ranker.z[v] @ ranker.z[head]), v))[:k]
        for name, ranked in [('gnn_exact', exact), ('gnn_voyager', approx),
                             ('gnn_no_relation', no_relation), ('graph_bfs', bfs)]:
            if len(ranked) != k or len(set(ranked)) != k or not set(ranked).issubset(set(candidates)):
                raise ValueError(f'Invalid ranking returned by {name}')
            result = _measures(ranked, positives, ks)
            scores[name].append(result)
            rows.append({'method': name, 'source': graph.ids[head],
                         'relation': graph.relation_names[rel], 'n_candidates': len(candidates),
                         'n_relevant': len(positives), 'run': 0, **result})
        for trial in range(random_repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, head, rel, trial]))
            choice = rng.permutation(candidates).tolist()[:k]
            result = _measures(choice, positives, ks)
            scores['random'].append(result)
            rows.append({'method': 'random', 'source': graph.ids[head],
                         'relation': graph.relation_names[rel], 'n_candidates': len(candidates),
                         'n_relevant': len(positives), 'run': trial, **result})
    if not scores['gnn_exact']:
        raise ValueError('No supported held-out relation queries')
    metrics = [f'{name}@{k}' for k in ks for name in ('hits', 'recall', 'mrr')]
    results = {name: {'metrics': {metric: float(np.mean([s[metric] for s in entries]))
                                 for metric in metrics}, 'query_evaluations': len(entries)}
               for name, entries in scores.items()}
    report = {'task': 'held_out_typed_triple_retrieval', 'query_count': len(scores['gnn_exact']),
              'test_triples': len(split.test), 'grouped_pairs': split.grouped_pairs,
              'excluded_validation_targets': 'evaluation filter only (not training)',
              'candidate_rule': 'relation-compatible target node kind, excluding training triples and self',
              'query_rule': '(head, relation) -> rank eligible tails',
              'scoring': 'cosine(normalize(z_head + relation_vector), z_tail)',
              'negative_sampling': 'relation typed; no validation/test positive access during fitting',
              'seed': seed, 'ks': list(ks), 'random_repeats': random_repeats,
              'skipped_queries': skipped, 'methods': results,
              'limitations': [
                  'v3 node-pair metrics are NOT directly comparable: different task and candidate sets',
                  'One transductive split; descriptive scores only, no significance claim',
                  'GraphSAGE encoder aggregates all training relation types equally',
                  'Unobserved links are not guaranteed negative; filtered evaluation uses validation labels',
                  'gnn_no_relation disables query translation only, not a separately trained v3 model',
                  'AUC is not the principal objective: relation-specific ranking is evaluated here',
                  'KG triple discovery, not biomedical fact validation or LLM answer correctness',
              ]}
    return report, rows


def markdown_report(report: dict) -> str:
    lines = ['# Relation-aware held-out triple retrieval (v4)', '',
             f"- Queries (head, relation): {report['query_count']}",
             f"- Held-out triples: {report['test_triples']}",
             f"- Candidate rule: {report['candidate_rule']}",
             f"- Random repeats: {report['random_repeats']}; seed: {report['seed']}",
             '', '| Method | K | Hits@K | Recall@K | MRR@K |', '|---|---:|---:|---:|---:|']
    for method, item in report['methods'].items():
        for k in report['ks']:
            m = item['metrics']
            lines.append(f"| {method} | {k} | {m[f'hits@{k}']:.3f} | {m[f'recall@{k}']:.3f} | {m[f'mrr@{k}']:.3f} |")
    lines += ['', '## Interpretation boundaries', '']
    lines += ['- ' + note for note in report['limitations']]
    return '\n'.join(lines) + '\n'
