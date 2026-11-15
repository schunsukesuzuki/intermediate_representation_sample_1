"""No Polars/Voyager required: deterministic algorithm-level regression tests."""
from dataclasses import dataclass

import jax
import numpy as np

from gnn_rag.triple_learning import (triple_split, typed_candidate_ids,
    candidate_without_observed, sample_typed_negatives, name_hash_features,
    train_triples)
from gnn_rag.typed_search import RelationRanker
from gnn_rag.triple_evaluation import evaluate_triples, markdown_report


class Column:
    def __init__(self, values): self.values = values
    def to_list(self): return list(self.values)


class Nodes:
    def __init__(self, names, topics): self.cols = {'name': names, 'topic': topics}
    def get_column(self, key): return Column(self.cols[key])


@dataclass
class MiniGraph:
    ids: tuple
    node_kinds: tuple
    relation_names: tuple
    senders: np.ndarray
    receivers: np.ndarray
    relation_ids: np.ndarray
    weights: np.ndarray
    nodes: Nodes
    features: np.ndarray

    @property
    def n_nodes(self): return len(self.ids)


def graph_fixture():
    ids = tuple(f'd{i:02}' for i in range(12)) + tuple(f'g{i:02}' for i in range(18))
    kinds = ('Disease',)*12 + ('Gene',)*18
    # Relations: DrD, DaG and GiG; use the official target type for each.
    rel_names = ('DaG','DrD','GiG')
    edges = []
    for h in range(12):
        for j in range(5):
            edges.append((h, 0, 12 + ((h * 3 + j * 2) % 18)))
        edges.append((h, 1, (h + 1) % 12))
        edges.append((h, 1, (h + 3) % 12))
    for g in range(18):
        edges.append((12 + g, 2, 12 + ((g + 1) % 18)))
        edges.append((12 + g, 2, 12 + ((g + 5) % 18)))
    edges = sorted(set(edges))
    a = np.asarray(edges, dtype=np.int32)
    names = tuple(f'Disease {i} neuro' for i in range(12)) + tuple(f'Gene {i} protein' for i in range(18))
    return MiniGraph(ids, kinds, rel_names, a[:, 0], a[:, 2], a[:, 1],
                     np.ones(len(a), dtype=np.float32), Nodes(names, kinds),
                     np.ones((30, 2), dtype=np.float32))


def test_split_prevents_same_pair_cross_partition_leakage():
    g = graph_fixture()
    sp = triple_split(g, seed=42)
    def pairs(a): return {tuple(sorted((int(s), int(t)))) for s, _, t in a}
    assert pairs(sp.train).isdisjoint(pairs(sp.test))
    assert pairs(sp.train).isdisjoint(pairs(sp.validation))
    assert pairs(sp.test).isdisjoint(pairs(sp.validation))
    assert len(sp.train) + len(sp.validation) + len(sp.test) == len(g.senders)
    assert np.array_equal(sp.train, triple_split(g, seed=42).train)


def test_negatives_are_relation_typed_and_no_training_positives():
    g = graph_fixture()
    sp = triple_split(g)
    known = {tuple(t) for t in sp.train.tolist()}
    negatives = sample_typed_negatives(g, sp.train, sp.train, np.random.default_rng(7))
    assert len(negatives)
    for h, r, t in negatives:
        assert tuple([int(h), int(r), int(t)]) not in known
        assert g.node_kinds[t] == {'DaG': 'Gene', 'DrD': 'Disease', 'GiG': 'Gene'}[g.relation_names[r]]
        assert h != t
    np.testing.assert_array_equal(typed_candidate_ids(g, 0, 0), np.arange(12, 30))


def test_name_features_not_only_node_type_and_seed_fixed():
    g = graph_fixture()
    x = name_hash_features(g)
    assert x.shape == (g.n_nodes, 2 + 64)
    assert not np.allclose(x[0], x[1])
    np.testing.assert_array_equal(x, name_hash_features(g))


def test_relation_vectors_train_and_ranking_metrics_are_computable():
    g = graph_fixture()
    split = triple_split(g, seed=42)
    result = train_triples(g, split, seed=42, epochs=5, hidden_dim=12, embed_dim=8)
    assert np.isfinite(result.embeddings).all()
    assert np.isfinite(result.metrics['train_loss'])
    assert result.metrics['n_relations'] == 3
    vectors = np.asarray(result.params['relation_vectors'])
    assert vectors.shape == (3, 8)
    ranker = RelationRanker(g.ids, g.node_kinds, g.relation_names, result.embeddings, vectors)
    head, rel = map(int, split.test[0][:2])
    candidates = candidate_without_observed(g, head, rel, split.train)
    assert all(g.node_kinds[t] == ({'DaG':'Gene','DrD':'Disease','GiG':'Gene'}[g.relation_names[rel]]) for t in candidates)
    ranked = ranker.exact(head, rel, candidates)
    assert sorted(ranked) == sorted(candidates.tolist())
    # When the approximate ranking is deliberately replaced by exact scores,
    # the entire benchmark must report identical model metrics.
    ranker.voyager = lambda h, r, c, k: ranker.exact(h, r, c, k)
    report, rows = evaluate_triples(g, split, ranker, random_repeats=2, seed=42)
    assert report['query_count'] > 0
    assert report['methods']['gnn_exact']['metrics'] == report['methods']['gnn_voyager']['metrics']
    assert '| gnn_exact |' in markdown_report(report)
    assert rows


def test_two_relations_same_pair_are_preserved():
    g = graph_fixture()
    # Duplicate a node pair with another relation (synthetic diagnostic only).
    g.senders = np.append(g.senders, g.senders[0])
    g.receivers = np.append(g.receivers, g.receivers[0])
    g.relation_ids = np.append(g.relation_ids, 1)
    g.weights = np.append(g.weights, 1.)
    sp = triple_split(g)
    pair = tuple(sorted((int(g.senders[0]), int(g.receivers[0]))))
    partitions = [p for p in (sp.train, sp.validation, sp.test)
                  if any(tuple(sorted((int(s),int(t)))) == pair for s,_,t in p)]
    assert len(partitions) == 1
