import numpy as np
from gnn_rag.fast_evaluation import QR, fused, measure, choose_queries
from gnn_rag.fusion_evaluation import QueryRanks, fused_ranking


def test_fast_fusion_matches_reference_across_ranks():
    rng=np.random.default_rng(2026)
    for size in (1, 2, 7, 100, 800):
        choices=np.sort(rng.choice(1500,size=size,replace=False)).astype(np.int32)
        first=rng.permutation(choices)
        second=rng.permutation(choices)
        q_fast=QR(0,0,choices,frozenset({int(choices[-1])}),first,second)
        q_reference=QueryRanks(0,0,tuple(choices.tolist()),frozenset({int(choices[-1])}),
                               tuple(first.tolist()),tuple(second.tolist()))
        for alpha in (0.0, 0.05, 0.2, 0.5, 0.75, 0.95, 1.0):
            assert fused(q_fast,alpha).tolist()==list(fused_ranking(q_reference,alpha))


def test_fast_measure_is_valid():
    q=QR(0,0,np.array([1,2,3],dtype=np.int32),frozenset({3}),
         np.array([2,3,1],dtype=np.int32),np.array([1,3,2],dtype=np.int32))
    assert measure(fused(q,1.0),q.positives,(1,3))['mrr@3']==0.5
