"""Leakage, deterministic ranking, and calibration tests (no Polars/Voyager)."""
from dataclasses import replace

import numpy as np
import pytest

from gnn_rag.fusion_evaluation import (
    QueryRanks, fused_ranking, calibrate_alpha, compare_three,
    make_query_ranks, rank_candidates, markdown_comparison,
)
from gnn_rag.triple_learning import triple_split, train_triples
from gnn_rag.typed_search import RelationRanker

# The test graph is generated locally; no downloaded or held-out records.
from test_v4_triple_core import graph_fixture


def _fixture():
    graph = graph_fixture()
    split = triple_split(graph, seed=42)
    trained = train_triples(graph, split, seed=42, epochs=3, hidden_dim=12, embed_dim=8)
    ranker = RelationRanker(graph.ids, graph.node_kinds, graph.relation_names,
                            trained.embeddings, np.asarray(trained.params['relation_vectors']))
    return graph, split, ranker


def test_fusion_score_endpoints_exactly_reproduce_baselines():
    q = QueryRanks(0, 0, (4, 5, 6, 7), frozenset({4}),
                   (7, 6, 5, 4), (4, 6, 5, 7))
    assert fused_ranking(q, 0.) == q.bfs
    assert fused_ranking(q, 1.) == q.gnn
    assert set(fused_ranking(q, .5)) == set(q.candidates)
    assert fused_ranking(q, .5) == fused_ranking(q, .5)
    with pytest.raises(ValueError): fused_ranking(q, 1.2)


def test_calibration_only_uses_validation_query_objects():
    q = QueryRanks(0, 0, (1, 2, 3), frozenset({1}), (3, 2, 1), (1, 2, 3))
    best = calibrate_alpha([q], objective_k=3)
    assert best['alpha'] == 0.0
    assert best['fallback_to_bfs'] is True
    assert best['uses_test_labels'] is False


def test_three_methods_same_candidates_and_heldout_query_rows():
    graph, split, ranker = _fixture()
    report, rows = compare_three(graph, split, ranker, seed=42)
    assert report['calibration']['uses_test_labels'] is False
    assert report['query_count'] > 0
    assert set(report['methods']) == {'gnn_only', 'bfs_only', 'bfs_gnn'}
    assert len(rows) == 3 * report['query_count']
    assert '| bfs_only |' in markdown_comparison(report)
    assert '| bfs_gnn |' in markdown_comparison(report)
    for chunk in (rows[i:i+3] for i in range(0, len(rows), 3)):
        assert {r['method'] for r in chunk} == set(report['methods'])
        assert len({r['n_candidates'] for r in chunk}) == 1
        assert len({r['n_relevant'] for r in chunk}) == 1
        assert len({(r['source'], r['relation']) for r in chunk}) == 1
    assert min(report['candidate_counts_test'].values()) >= 1


def test_test_labels_do_not_change_tuned_weight():
    graph, split, ranker = _fixture()
    original_val = make_query_ranks(graph, split, ranker, partition='validation')
    original_alpha = calibrate_alpha(original_val)['alpha']
    # Deliberately alter held-out TEST labels while leaving TRAIN, VALIDATION,
    # features, model parameters and candidate inputs unchanged.
    corrupted_test = split.test.copy()
    corrupted_test[:, 2] = (corrupted_test[:, 2] + 1) % graph.n_nodes
    altered = replace(split, test=corrupted_test)
    new_val = make_query_ranks(graph, altered, ranker, partition='validation')
    assert original_alpha == calibrate_alpha(new_val)['alpha']


def test_inference_ranks_match_validation_bfs_and_gnn():
    graph, split, ranker = _fixture()
    queries = make_query_ranks(graph, split, ranker, partition='validation')
    q = queries[0]
    gnn = rank_candidates(ranker, split.train, q.head, q.relation,
                          np.asarray(q.candidates), alpha=.3, method='gnn_only')
    bfs = rank_candidates(ranker, split.train, q.head, q.relation,
                          np.asarray(q.candidates), alpha=.3, method='bfs_only')
    assert gnn == list(q.gnn)
    assert bfs == list(q.bfs)
    assert len(rank_candidates(ranker, split.train, q.head, q.relation,
                               np.asarray(q.candidates), alpha=.3, method='bfs_gnn')) == len(q.candidates)
    with pytest.raises(ValueError):
        rank_candidates(ranker, split.train, q.head, q.relation,
                        np.asarray(q.candidates), alpha=.3, method='bogus')


def test_bfs_uses_training_graph_only_not_hidden_links():
    graph, split, ranker = _fixture()
    rows = make_query_ranks(graph, split, ranker, partition='test')
    # The baseline must be identical with an independently computed rank
    # via TRAIN graph. This also catches accidentally passing validation/test.
    q = rows[0]
    assert rank_candidates(ranker, split.train, q.head, q.relation,
                           np.asarray(q.candidates), alpha=0., method='bfs_only') == list(q.bfs)
