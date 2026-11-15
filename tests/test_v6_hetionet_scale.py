"""Reproducible 10k-scale preparation and leakage guards without real KG download.

Synthetic fixtures only test functionality, NOT prediction performance.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from gnn_rag.hetionet_scale import convert_full_hetionet, load_full_hetionet, sample_indices
from gnn_rag.compose_prepare import cached_scaled_hetionet_is_usable
from gnn_rag.triple_learning import (CachedTypedNegativeSampler,
                                      expand_symmetric_training, triple_split, train_triples)
from gnn_rag.typed_search import RelationRanker
from gnn_rag.fusion_evaluation import compare_three, make_query_ranks
from test_v4_triple_core import graph_fixture


def _write_raw(tmp: Path, *, n: int = 250, m: int = 2000) -> tuple[Path, Path]:
    rng = np.random.default_rng(71)
    nodes = tmp / 'hetionet-v1.0-nodes.tsv'
    edges = tmp / 'hetionet-v1.0-edges.sif.gz'
    with nodes.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(['id', 'name', 'kind'])
        for i in range(n):
            writer.writerow([f'Gene::{i}', f'Natural gene {i}', 'Gene'])
    relations = ['GiG', 'GcG']
    triples = set()
    while len(triples) < m:
        a, b = rng.choice(n, 2, replace=False)
        triples.add((f'Gene::{a}', relations[len(triples) % 2], f'Gene::{b}'))
    with gzip.open(edges, 'wt', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(['source', 'metaedge', 'target'])
        writer.writerows(sorted(triples))
    return nodes, edges


def test_full_source_has_reproducible_typed_exact_budget(tmp_path):
    nodes, edges = _write_raw(tmp_path)
    out = tmp_path / 'prepared'
    meta = convert_full_hetionet(nodes, edges, out,
                                 target_triples=500, max_nodes=200, seed=41,
                                 focus='Natural gene 0')
    assert meta['schema_version'] == 6
    assert meta['edges'] == 500
    assert meta['nodes'] <= 200
    assert meta['full_source_triple_count'] == 2000
    assert meta['triple_preserving'] is True
    assert meta['selection_based_on_full_unsplit_graph'] is True
    assert cached_scaled_hetionet_is_usable(out, target=500, max_nodes=200, seed=41)[0]
    with (out / 'edges.csv').open(newline='') as fh:
        output_triples = [(x['source'], x['relation'], x['target']) for x in csv.DictReader(fh)]
    assert len(set(output_triples)) == 500
    first = set(output_triples)
    convert_full_hetionet(nodes, edges, out, target_triples=500, max_nodes=200,
                          seed=41, focus='Natural gene 0')
    with (out / 'edges.csv').open(newline='') as fh:
        assert {(x['source'], x['relation'], x['target']) for x in csv.DictReader(fh)} == first
    assert not cached_scaled_hetionet_is_usable(out, target=501, max_nodes=200, seed=41)[0]
    assert not cached_scaled_hetionet_is_usable(out, target=500, max_nodes=201, seed=41)[0]


def test_sample_indices_runs_with_target_10k_without_name_or_clan_features():
    rng = np.random.default_rng(4)
    n = 1800
    src = rng.integers(0, n, size=60_000, dtype=np.int32)
    dst = rng.integers(0, n, size=60_000, dtype=np.int32)
    rel = rng.integers(0, 3, size=60_000, dtype=np.int32)
    ids = tuple(f'Node::{i}' for i in range(n))
    kinds = ['Gene' if i % 2 else 'Disease' for i in range(n)]
    nodes, rows, stats = sample_indices(ids, kinds, src, dst, rel,
        target_triples=10_000, max_nodes=1600, seed=41, focus_index=0)
    assert len(rows) == 10_000
    assert len(nodes) <= 1600
    assert 0 in nodes
    assert stats['spanning_forest_triples'] <= len(nodes) - 1
    assert len({(int(src[i]), int(rel[i]), int(dst[i])) for i in rows}) == 10000


def test_cached_typed_sampler_excludes_only_observed_training_edges():
    g = graph_fixture()
    sp = triple_split(g, seed=11)
    train = expand_symmetric_training(g, sp.train)
    sampler = CachedTypedNegativeSampler(g, train)
    pos, negatives = sampler.draw(train[:30], np.random.default_rng(1), 4)
    known = set(map(tuple, train.tolist()))
    assert len(pos) == len(negatives)
    for row in negatives.reshape(-1, 3):
        h, r, t = map(int, row)
        assert (h, r, t) not in known
        assert t != h
        assert g.node_kinds[t] == {'DaG': 'Gene', 'DrD': 'Disease', 'GiG': 'Gene'}[g.relation_names[r]]


def test_early_stopping_budgets_do_not_influence_held_out_labels():
    g = graph_fixture()
    split = triple_split(g, seed=42)
    trained = train_triples(g, split, seed=42, epochs=2,
                           max_positives_per_epoch=12, feature_mode='kind_only')
    assert trained.metrics['max_ranking_positives_per_epoch'] == 12
    ranker = RelationRanker(g.ids, g.node_kinds, g.relation_names,
                            trained.embeddings, np.asarray(trained.params['relation_vectors']))
    val_queries = make_query_ranks(g, split, ranker, partition='validation', max_queries=5, seed=9)
    assert len(val_queries) == 5
    assert len({(q.head, q.relation) for q in val_queries}) == 5
    report, rows = compare_three(g, split, ranker, max_queries=5, seed=9)
    assert report['query_count'] == 5
    assert report['validation_query_count'] == 5
    assert len(rows) == 15
    assert report['calibration']['uses_test_labels'] is False
