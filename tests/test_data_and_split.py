import numpy as np
import polars as pl
import pytest

from gnn_rag.data import load_graph
from gnn_rag.split import split_edges


def test_polars_id_mapping_and_feature_shapes(graph):
    assert graph.n_nodes == 20
    assert graph.features.shape[0] == graph.n_nodes
    assert np.all(graph.features.sum(axis=1) == 2.0)
    assert graph.id_to_index["n00"] == 0
    assert len(graph.positive_edges) == len(graph.senders)


def test_splits_do_not_leak_positive_edges_into_training_graph(graph):
    split = split_edges(graph, seed=7)
    train = set(map(tuple, split.train.tolist()))
    validation = set(map(tuple, split.validation.tolist()))
    test = set(map(tuple, split.test.tolist()))
    negs = set(map(tuple, split.validation_neg.tolist())) | set(map(tuple, split.test_neg.tolist()))
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert negs.isdisjoint(graph.positive_edges)
    assert all((t, s) not in graph.positive_edges for s, t in negs)
    assert all((t, s) not in train for s, t in validation | test)
    assert all(s != t for s, t in negs)
    assert len(split.train_weight) == len(split.train)


def test_split_seed_is_reproducible(graph):
    a, b = split_edges(graph, seed=5), split_edges(graph, seed=5)
    np.testing.assert_array_equal(a.train, b.train)
    np.testing.assert_array_equal(a.test_neg, b.test_neg)


def test_validation_rejects_duplicate_ids(graph, tmp_path):
    nodes = pl.concat([graph.nodes, graph.nodes.head(1)])
    nodes.write_csv(tmp_path / "nodes.csv")
    graph.edges.write_csv(tmp_path / "edges.csv")
    with pytest.raises(ValueError, match="Duplicate node_id"):
        load_graph(tmp_path / "nodes.csv", tmp_path / "edges.csv")
