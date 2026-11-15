"""ALGORITHM-ONLY 10k triple smoke test; NOT a Hetionet benchmark.

No Polars, Voyager, or downloaded files are needed; JAX/NumPy are used.
The node features contain ONLY public node types, no hidden cluster labels.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from gnn_rag.fusion_evaluation import compare_three
from gnn_rag.triple_learning import triple_split, train_triples
from gnn_rag.typed_search import RelationRanker


class FakeColumn:
    def __init__(self, values): self.values = values
    def to_list(self): return list(self.values)


class FakeNodes:
    def __init__(self, names, kinds):
        self._cols = {'name': names, 'topic': kinds}
    def get_column(self, key): return FakeColumn(self._cols[key])


@dataclass
class FakeGraph:
    ids: tuple
    node_kinds: tuple
    relation_names: tuple
    senders: np.ndarray
    receivers: np.ndarray
    relation_ids: np.ndarray
    weights: np.ndarray
    nodes: FakeNodes
    features: np.ndarray
    @property
    def n_nodes(self): return len(self.ids)


def make_graph(n_disease: int, n_gene: int, triples: int, seed: int) -> FakeGraph:
    rng = np.random.default_rng(seed)
    d, n = n_disease, n_disease+n_gene
    seen = set()
    while len(seen) < triples:
        rel = int(rng.integers(3))
        if rel == 0:
            h, t = int(rng.integers(d)), int(rng.integers(d, n))
        elif rel == 1:
            h, t = int(rng.integers(d)), int(rng.integers(d))
        else:
            h, t = int(rng.integers(d,n)), int(rng.integers(d,n))
        if h != t:
            seen.add((h, rel, t))
    a = np.asarray(sorted(seen), dtype=np.int32)
    ids = tuple(f'Disease::{i}' for i in range(d)) + tuple(f'Gene::{i}' for i in range(n_gene))
    kinds = ('Disease',)*d + ('Gene',)*n_gene
    names = ('disease',)*d + ('gene',)*n_gene  # no hidden or ID-based text features
    return FakeGraph(ids, kinds, ('DaG','DrD','GiG'), a[:,0], a[:,2], a[:,1],
                     np.ones(len(a),dtype=np.float32), FakeNodes(names,kinds),
                     np.zeros((n,2),dtype=np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--triples', type=int, default=10000)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-queries', type=int, default=8)
    p.add_argument('--max-positives', type=int, default=2048)
    p.add_argument('--output', type=Path)
    a=p.parse_args()
    start = time.monotonic()
    g=make_graph(400,600,a.triples,19)
    sp=triple_split(g,seed=a.seed)
    result=train_triples(g,sp,seed=a.seed,epochs=a.epochs,
        feature_mode='kind_only',max_positives_per_epoch=a.max_positives)
    r=RelationRanker(g.ids,g.node_kinds,g.relation_names,result.embeddings,
                     np.asarray(result.params['relation_vectors']))
    report,_=compare_three(g,sp,r,max_queries=a.eval_queries,seed=a.seed)
    output = {
        'data_type': 'synthetic functionality only — NOT Hetionet',
        'n_nodes': g.n_nodes, 'triples': a.triples, 'train_triples':len(sp.train),
        'validation_triples':len(sp.validation),'test_triples':len(sp.test),
        'feature_mode':'kind_only','epochs':a.epochs,
        'seed':a.seed,'evaluated_test_queries':report['query_count'],
        'alpha':report['calibration']['alpha'],
        'test_mrr10':{k:v['test']['mrr@10'] for k,v in report['methods'].items()},
        'elapsed_seconds_this_machine':round(time.monotonic()-start,2),
        'purpose': 'Verify one 10k-triple JAX+ranking execution; scores are not a biomedical result',
    }
    if a.output:
        a.output.parent.mkdir(exist_ok=True, parents=True)
        a.output.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
