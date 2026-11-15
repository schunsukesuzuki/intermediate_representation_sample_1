"""Offline checks for converter semantics, cache migration, and Voyager API shape.
No real GitHub download, Polars installation, or Voyager binary needed.
"""
import csv
import gzip
import importlib
import json
import sys
import types

import numpy as np
from gnn_rag import compose_prepare
from gnn_rag.typed_search import RelationRanker


def test_v3_cache_is_rebuilt_even_when_csv_exists(tmp_path):
    folder = tmp_path / 'hetionet_demo'
    folder.mkdir()
    (folder / 'nodes.csv').write_text('node_id,name,kind\nx,x,Disease\n', encoding='utf-8')
    (folder / 'edges.csv').write_text('source,target,relation\nx,y,DrD\n', encoding='utf-8')
    (folder / 'dataset_meta.json').write_text(json.dumps({'focus_name': "Alzheimer's disease"}), encoding='utf-8')
    valid, reason = compose_prepare.cached_hetionet_is_usable(folder,
        focus="Alzheimer's disease", max_nodes=120, seed=42)
    assert not valid and 'old pair-collapsed cache' in reason


def test_hetio_converter_preserves_multiple_relation_triples(tmp_path, monkeypatch):
    # Replace only the Polars CSV writer in this isolated test process.
    class DataFrame:
        def __init__(self, rows): self.rows = rows
        def write_csv(self, path):
            with open(path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
    monkeypatch.setitem(sys.modules, 'polars', types.SimpleNamespace(DataFrame=DataFrame))
    # Loader is pure standard-library apart from the adapter's import-time pl.
    sys.modules.pop('gnn_rag.public_data', None)
    converter = importlib.import_module('gnn_rag.public_data')
    nodes_tsv = tmp_path / 'nodes.tsv'
    edges_sif = tmp_path / 'edges.sif.gz'
    names = [f'N{i:02}' for i in range(24)]
    with nodes_tsv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(['id', 'name', 'kind'])
        for i, name in enumerate(names):
            writer.writerow([name, "Alzheimer's disease" if i == 0 else name,
                             'Disease' if i < 12 else 'Gene'])
    with gzip.open(edges_sif, 'wt', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(['source', 'metaedge', 'target'])
        for i in range(1, 20): writer.writerow(['N00', 'DaG', names[i]])
        for i in range(1, 20): writer.writerow(['N00', 'DrD', names[i]])
        writer.writerow(['N00', 'DrD', 'N01'])  # duplicate exact triple must be deduplicated
    result = converter.convert_hetionet(nodes_tsv, edges_sif, tmp_path/'out', max_nodes=24)
    assert result['schema_version'] == 4 and result['triple_preserving']
    with (tmp_path/'out/edges.csv').open() as fh:
        triples = {(r['source'], r['relation'], r['target']) for r in csv.DictReader(fh)}
    assert any((s,'DaG',t) in triples and (s,'DrD',t) in triples for s,_,t in triples)
    assert len(triples) == result['edges']
    sys.modules.pop('gnn_rag.public_data', None)


def test_relation_conditioned_voyager_uses_same_score_as_exact(monkeypatch):
    class Index:
        def __init__(self, space, num_dimensions, max_elements, random_seed):
            self.values = []
        def add_item(self, vector):
            self.values.append(np.asarray(vector))
            return len(self.values)-1
        def query(self, vector, k):
            values = np.asarray(self.values)
            d = np.linalg.norm(values-vector, axis=1)
            indices = np.argsort(d, kind='stable')[:k]
            return indices, d[indices]
    monkeypatch.setitem(sys.modules, 'voyager', types.SimpleNamespace(Index=Index,
                                                        Space=types.SimpleNamespace(Euclidean='euclidean')))
    names = ('D0', 'D1', 'G0', 'G1', 'G2')
    kinds = ('Disease', 'Disease', 'Gene','Gene','Gene')
    z = np.asarray([[1,0], [0,1], [.9,.1], [.2,.8], [-1,0]], dtype=np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    rel = np.asarray([[0,0]], dtype=np.float32)
    ranker = RelationRanker(names, kinds, ('DaG',), z, rel)
    expected = ranker.exact(0, 0, np.asarray([2, 3, 4]), 2)
    assert ranker.voyager(0, 0, np.asarray([2, 3, 4]), 2) == expected
    assert ranker.voyager(0, 0, np.asarray([3, 4]), 2) == ranker.exact(0, 0, np.asarray([3, 4]), 2)
