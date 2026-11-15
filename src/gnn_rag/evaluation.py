"""Leakage-controlled, transductive, node-to-node held-out link retrieval.

For every test source node, rank *unobserved* candidate neighbors using:
- GNN trained vectors + Voyager ANN (not an exact dot-product oracle);
- two-/multi-hop BFS on training edges only (ties broken by train degree);
- uniform random permutations with repeated independently seeded trials.

Evaluate Hits@K / Recall@K / MRR@K over test neighbors only.
This tests held-out link discovery, NOT relevance of generated answers.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QueryEvaluation:
    source: int
    targets: frozenset[int]
    candidates: tuple[int, ...]


def build_queries(n_nodes: int, train: np.ndarray, test: np.ndarray) -> tuple[list[set[int]], list[QueryEvaluation]]:
    if n_nodes < 3:
        raise ValueError("At least three nodes are required")
    neighbors = [set() for _ in range(n_nodes)]
    for src, dst in np.asarray(train, dtype=np.int64).reshape(-1, 2):
        if src == dst or min(src, dst) < 0 or max(src, dst) >= n_nodes:
            raise ValueError("Invalid training edge")
        neighbors[int(src)].add(int(dst))
        neighbors[int(dst)].add(int(src))
    targets = [set() for _ in range(n_nodes)]
    for src, dst in np.asarray(test, dtype=np.int64).reshape(-1, 2):
        if src == dst or min(src, dst) < 0 or max(src, dst) >= n_nodes:
            raise ValueError("Invalid test edge")
        if dst in neighbors[int(src)] or src in neighbors[int(dst)]:
            raise ValueError("Held-out positive is already visible in training adjacency")
        targets[int(src)].add(int(dst))
        targets[int(dst)].add(int(src))
    queries = []
    for source, positives in enumerate(targets):
        if not positives:
            continue
        candidates = tuple(i for i in range(n_nodes) if i != source and i not in neighbors[source])
        if not positives.issubset(candidates):
            raise ValueError("Held-out test links not in candidate set")
        queries.append(QueryEvaluation(source, frozenset(positives), candidates))
    if not queries:
        raise ValueError("No test positives for link retrieval evaluation")
    return neighbors, queries


def graph_rank(query: QueryEvaluation, adjacency: list[set[int]]) -> list[int]:
    """BFS shortest train-only distance; no test-edge information for ranking."""
    source = query.source
    degree = [len(n) for n in adjacency]
    distance = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in sorted(adjacency[u]):
            if v not in distance:
                distance[v] = distance[u] + 1
                queue.append(v)
    return sorted(query.candidates, key=lambda v: (distance.get(v, float('inf')), -degree[v], v))


def _scores(rank: list[int], positives: frozenset[int], ks: tuple[int, ...]) -> dict:
    result = {}
    for k in ks:
        top = rank[:k]
        matches = [i + 1 for i, v in enumerate(top) if v in positives]
        result[f"hits@{k}"] = float(bool(matches))
        result[f"recall@{k}"] = len(matches) / len(positives)
        result[f"mrr@{k}"] = 1.0 / min(matches) if matches else 0.0
    return result


def compare_retrieval(
    node_ids: tuple[str, ...], train: np.ndarray, test: np.ndarray, retriever: Any,
    *, ks: tuple[int, ...] = (1, 3, 5, 10), random_repeats: int = 20, seed: int = 42,
) -> tuple[dict, list[dict]]:
    """Return report and per-query diagnostic rows. Test labels never rank candidates."""
    if not ks or min(ks) < 1 or random_repeats < 1:
        raise ValueError("Positive ks and random_repeats are required")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Node IDs must be unique")
    ks = tuple(sorted(set(int(k) for k in ks)))
    neighbors, queries = build_queries(len(node_ids), train, test)
    max_k = max(ks)
    rows = []
    per_method: dict[str, list[dict]] = {"gnn_voyager": [], "graph_bfs": [], "random": []}
    for q in queries:
        excluded_ids = frozenset(node_ids[v] for v in neighbors[q.source])
        # VoyagerRetriever.nearest already excludes self; request K candidates after filters.
        # We evaluate truncated @K only to avoid a full-index HNSW scan.
        approx = retriever.nearest(node_ids[q.source], top_k=min(len(q.candidates), max_k),
                                   exclude=excluded_ids)
        gnn_rank = [node_ids.index(n.node_id) for n in approx]
        if len(set(gnn_rank)) != len(gnn_rank) or any(v not in q.candidates for v in gnn_rank):
            raise ValueError("Voyager returned invalid or excluded candidate(s)")
        if len(gnn_rank) < min(max_k, len(q.candidates)):
            raise ValueError("Voyager returned insufficient candidates")
        bfs_rank = graph_rank(q, neighbors)
        for method, rank in (("gnn_voyager", gnn_rank), ("graph_bfs", bfs_rank)):
            results = _scores(rank, q.targets, ks)
            per_method[method].append(results)
            rows.append({"method": method, "source": node_ids[q.source],
                         "num_relevant": len(q.targets), "run": 0, **results})
        # Repeated random trials make a one-seed random baseline less noisy.
        for repeat in range(random_repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, q.source, repeat]))
            rank = list(rng.permutation(np.asarray(q.candidates, dtype=np.int32)))
            results = _scores(rank, q.targets, ks)
            per_method["random"].append(results)
            rows.append({"method": "random", "source": node_ids[q.source],
                         "num_relevant": len(q.targets), "run": repeat, **results})
    metric_names = [f"{metric}@{k}" for k in ks for metric in ("hits", "recall", "mrr")]
    methods = {}
    for method, evaluations in per_method.items():
        means = {metric: float(np.mean([r[metric] for r in evaluations]))
                 for metric in metric_names}
        methods[method] = {"metrics": means, "query_evaluations": len(evaluations)}
        if method == "random":
            # Std across run-level macro means (NOT confidence intervals).
            trial_means = {metric: [
                float(np.mean([rowscore[metric] for rowscore in evaluations[repeat::random_repeats]]))
                for repeat in range(random_repeats)] for metric in metric_names}
            methods[method]["run_std"] = {metric: float(np.std(vals, ddof=1))
                                           if len(vals) > 1 else 0.0
                                           for metric, vals in trial_means.items()}
    report = {
        "task": "held_out_node_neighbor_retrieval",
        "methodology": "transductive: train adjacency and features only; test links withheld until scoring",
        "query_nodes": len(queries),
        "held_out_undirected_pairs": len({tuple(sorted(map(int, p))) for p in np.asarray(test).reshape(-1, 2)}),
        "evaluated_relevant_endpoints": sum(len(q.targets) for q in queries),
        "candidate_rule": "all nodes except source and observed training neighbors",
        "metrics_definition": "macro over query nodes; random averages across independent trials; MRR@K reciprocal rank of first relevant node else 0",
        "ks": list(ks), "seed": seed, "random_repeats": random_repeats,
        "methods": methods,
        "limitations": [
            "Single transductive train/validation/test split; not a statistical significance claim",
            "Other positive edges absent from test labels may be counted as non-relevant",
            "ANN is approximate; no inference latency or HNSW speed benchmark measured",
            "GraphSAGE-style encoder is not relation-aware; types may dominate results",
            "This measures held-out link retrieval, not the quality of LLM answers or clinical truth",
        ],
    }
    return report, rows


def report_markdown(report: dict) -> str:
    ks = report["ks"]
    lines = ["# Held-out link retrieval benchmark", "",
             f"- Query nodes: {report['query_nodes']}",
             f"- Withheld undirected pairs: {report['held_out_undirected_pairs']}",
             f"- Random trials: {report['random_repeats']}; seed: {report['seed']}",
             "", "| Method | K | Hits@K | Recall@K | MRR@K |", "|---|---:|---:|---:|---:|"]
    for method, item in report["methods"].items():
        m = item["metrics"]
        for k in ks:
            lines.append(f"| {method} | {k} | {m[f'hits@{k}']:.3f} | "
                         f"{m[f'recall@{k}']:.3f} | {m[f'mrr@{k}']:.3f} |")
    lines += ["", "## Interpretation boundaries", ""]
    lines += [f"- {line}" for line in report["limitations"]]
    lines += ["", "Dataset and split-specific, not a clinical result. Lower or equal GNN scores "
              "are valid outcomes and must not be concealed.", ""]
    return "\n".join(lines)
