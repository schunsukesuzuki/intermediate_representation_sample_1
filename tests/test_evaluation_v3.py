"""Pure NumPy benchmark tests run without Polars/Voyager/Docker."""
from __future__ import annotations

import numpy as np
import pytest

from gnn_rag.evaluation import build_queries, compare_retrieval, graph_rank


class FakeRetriever:
    def __init__(self, ids):
        self.ids = ids

    def nearest(self, node_id, *, top_k, exclude):
        # Deliberate learned rank stub: strongest target preferentially indexed.
        ordered = list(reversed(self.ids))
        return [type("Neighbor", (), {"node_id": x})()
                for x in ordered if x != node_id and x not in exclude][:top_k]


def test_no_leakage_and_macro_metrics():
    ids = tuple(f"n{i}" for i in range(7))
    train = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int32)
    test = np.array([[0, 2], [5, 6]], dtype=np.int32)
    neighbors, queries = build_queries(len(ids), train, test)
    assert len(queries) == 4
    assert neighbors[0] == {1}
    q0 = [q for q in queries if q.source == 0][0]
    assert 1 not in q0.candidates
    assert 2 in q0.targets
    bfs = graph_rank(q0, neighbors)
    assert bfs[0] == 2  # 0→1→2 path in TRAIN, direct 0→2 test edge withheld.
    report, rows = compare_retrieval(ids, train, test, FakeRetriever(ids), ks=(1, 3),
                                     random_repeats=5, seed=41)
    assert report["query_nodes"] == 4
    assert report["held_out_undirected_pairs"] == 2
    assert report["methods"]["random"]["query_evaluations"] == 20
    assert len(rows) == (2 + 5) * 4
    assert 0 <= report["methods"]["gnn_voyager"]["metrics"]["mrr@3"] <= 1
    report2, rows2 = compare_retrieval(ids, train, test, FakeRetriever(ids), ks=(1, 3),
                                       random_repeats=5, seed=41)
    assert report == report2
    assert rows == rows2


def test_reversed_test_pair_is_detected_as_leakage():
    with pytest.raises(ValueError, match="visible"):
        build_queries(5, np.array([[0, 1]]), np.array([[1, 0]]))


def test_bad_settings_and_no_test():
    with pytest.raises(ValueError, match="No test"):
        build_queries(5, np.array([[0, 1]]), np.empty((0, 2), dtype=int))
    with pytest.raises(ValueError, match="Positive"):
        compare_retrieval(tuple(str(i) for i in range(5)), np.array([[0, 1]]),
                          np.array([[0, 2]]), FakeRetriever(tuple(str(i) for i in range(5))),
                          ks=(0, 3))
