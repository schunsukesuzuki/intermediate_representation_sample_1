"""Deterministic transductive link-prediction split.

Split reciprocal edges as one group to prevent leaked message-passing edges.
As our dot-product decoder is symmetric, negatives are unordered pairs absent
in both directions in the full observed graph. 'Unobserved' is NOT guaranteed
truly negative in a real KG: this is an illustrative closed-world assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import GraphData


@dataclass(frozen=True)
class EdgeSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    validation_neg: np.ndarray
    test_neg: np.ndarray
    train_neg_pool: np.ndarray
    train_weight: np.ndarray


def _canonical(edge: tuple[int, int]) -> tuple[int, int]:
    return tuple(sorted(edge))


def split_edges(graph: GraphData, seed: int = 42, validation_fraction: float = 0.15,
                test_fraction: float = 0.15) -> EdgeSplit:
    if not (0 < validation_fraction < 0.4 and 0 < test_fraction < 0.4):
        raise ValueError("Validation/test fractions must be in (0, 0.4)")
    rng = np.random.default_rng(seed)
    positives = np.column_stack([graph.senders, graph.receivers]).astype(np.int32)
    grouped: dict[tuple[int, int], list[int]] = {}
    for i, edge in enumerate(positives):
        grouped.setdefault(_canonical(tuple(int(x) for x in edge)), []).append(i)
    groups = list(grouped.values())
    n_groups = len(groups)
    n_val_groups = max(1, int(n_groups * validation_fraction))
    n_test_groups = max(1, int(n_groups * test_fraction))
    if n_groups - n_val_groups - n_test_groups < 2:
        raise ValueError("Too few distinct relation pairs")
    perm = rng.permutation(n_groups)

    def take(indices: np.ndarray) -> np.ndarray:
        return np.asarray([i for group_pos in indices for i in groups[group_pos]], dtype=np.int64)

    val_idx = take(perm[:n_val_groups])
    test_idx = take(perm[n_val_groups:n_val_groups + n_test_groups])
    train_idx = take(perm[n_val_groups + n_test_groups:])

    observed_unordered = set(grouped)
    candidates = np.asarray(
        [(s, t) for s in range(graph.n_nodes) for t in range(s + 1, graph.n_nodes)
         if (s, t) not in observed_unordered], dtype=np.int32,
    ).reshape(-1, 2)
    n_val, n_test = len(val_idx), len(test_idx)
    needed = n_val + n_test + len(train_idx)
    if len(candidates) < needed:
        raise ValueError("KG is too dense for disjoint negative sampling")
    rng.shuffle(candidates)
    val_neg = candidates[:n_val]
    test_neg = candidates[n_val:n_val + n_test]
    train_pool = candidates[n_val + n_test:]
    return EdgeSplit(
        train=positives[train_idx], validation=positives[val_idx],
        test=positives[test_idx], validation_neg=val_neg, test_neg=test_neg,
        train_neg_pool=train_pool, train_weight=graph.weights[train_idx],
    )
