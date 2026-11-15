#!/usr/bin/env python3
"""Integrity checks for reported 100k Hetionet results (does not train)."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from gnn_rag.triple_learning import triple_split

# Polars unavailable in the benchmark host. Pandas is an optional report-check
# adapter, same as in the portable src-less experiment runner.
from run_100k_real import load_csv_no_polars


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'))
    a=ap.parse_args();root=a.root
    data=root/'data/hetionet_demo'; res=root/'results'
    meta=json.loads((data/'dataset_meta.json').read_text())
    assert meta['nodes']==2330 and meta['edges']==100000 and len(meta['relation_counts'])==24
    g=load_csv_no_polars(data/'nodes.csv',data/'edges.csv')
    assert g.n_nodes==2330 and len(g.senders)==100000
    assert all('clan0' not in name.lower() for name in g.nodes.get_column('name').to_list())
    for seed in range(42,47):
        split=triple_split(g,seed=seed)
        group_sets=[]
        for rows in (split.train,split.validation,split.test):
            keys={(min(int(h),int(t)),max(int(h),int(t))) for h,_,t in rows}
            group_sets.append(keys)
        assert len(group_sets[0]&group_sets[1])==0
        assert len(group_sets[0]&group_sets[2])==0
        assert len(group_sets[1]&group_sets[2])==0
        assert sum(map(len,(split.train,split.validation,split.test)))==100000
        for mode in ('kind_name','kind_only'):
            report=json.loads((res/mode/f'evaluation_seed{seed}.json').read_text())
            assert report['query_count']==128
            assert report['calibration']['uses_test_labels'] is False
            assert 0<=report['calibration']['alpha']<=1
            assert (report['train_triples'],report['validation_triples'],report['test_triples']) == tuple(map(len,(split.train,split.validation,split.test)))
            npz=np.load(res/mode/f'gnn_embeddings_seed{seed}.npz')
            assert npz['embeddings'].shape==(2330,16) and npz['relation_vectors'].shape==(24,16)
            assert np.isfinite(npz['embeddings']).all() and np.isfinite(npz['relation_vectors']).all()
            with (res/mode/f'evaluation_per_query_seed{seed}.csv').open() as f:
                rows=list(csv.DictReader(f))
            assert len(rows)==384 # 3 methods * 128 queries
            comparisons={}
            for row in rows:
                key=(row['source'],row['relation'])
                value=(int(row['n_candidates']),int(row['n_relevant']))
                if key in comparisons:assert comparisons[key]==value
                comparisons[key]=value
            assert len(comparisons)==128
            for method in ('gnn_only','bfs_only','bfs_gnn'):
                sub=[r for r in rows if r['method']==method]
                assert len(sub)==128
                for m in ('hits@1','hits@10','recall@10','mrr@10'):
                    assert abs(np.mean([float(r[m]) for r in sub])-report['methods'][method]['test'][m])<1e-10
        print('PASS seed',seed,'pair-disjoint split, identical candidates, finite embeddings, metrics verified')
    print('ALL PASS')

if __name__=='__main__': main()
