"""Checks synthetic hidden group info cannot enter model features."""
from dataclasses import replace

import numpy as np

from gnn_rag.triple_learning import name_hash_features
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.offline_compare import Nodes, make_synthetic_kg


def test_kind_only_has_exactly_two_distinct_feature_rows():
    graph = make_synthetic_kg()
    x = name_hash_features(graph, feature_mode='kind_only')
    assert x.shape == (108, 2)
    assert len(np.unique(x, axis=0)) == 2
    assert np.all(x[:36] == x[0])
    assert np.all(x[36:] == x[36])
    assert not np.array_equal(x[0], x[36])


def test_kind_only_ignores_names_even_if_names_spoil_latent_group():
    graph = make_synthetic_kg()
    x = name_hash_features(graph, feature_mode='kind_only')
    polluted = tuple(f'Synthetic clan{i % 6}  /  {i}' for i in range(graph.n_nodes))
    changed = replace(graph, nodes=Nodes(polluted, graph.node_kinds))
    assert np.array_equal(x, name_hash_features(changed, feature_mode='kind_only'))


def test_scrambling_preserves_structure_but_removes_index_group_pattern():
    original = make_synthetic_kg()
    shuffled = make_synthetic_kg(scramble_indices=True)
    assert original.n_nodes == shuffled.n_nodes
    assert len(original.senders) == len(shuffled.senders) == 468
    assert np.array_equal(np.bincount(original.relation_ids), np.bincount(shuffled.relation_ids))
    # Every node still belongs to the same public node type.
    for h, t in zip(shuffled.senders, shuffled.receivers):
        assert 0 <= h < 108 and 0 <= t < 108
    assert np.array_equal(name_hash_features(original, feature_mode='kind_only'),
                          name_hash_features(shuffled, feature_mode='kind_only'))
    assert not np.array_equal(original.receivers, shuffled.receivers)


def test_clan_terms_not_present_in_node_names():
    graph = make_synthetic_kg()
    assert all('clan' not in name.lower() for name in graph.nodes.get_column('name').to_list())
    assert len(set(graph.nodes.get_column('name').to_list())) == 2
