#!/usr/bin/env python3
"""Summarize actual 100,000-triple benchmark outputs, without changing results."""
import argparse, json, csv
from pathlib import Path
import numpy as np

FEATURES=('kind_name','kind_only')
METHODS=('gnn_only','bfs_only','bfs_gnn')
METRICS=('hits@1','hits@10','recall@10','mrr@10')
SEEDS=tuple(range(42,47))

def build(root: Path):
    out={'dataset':'Hetionet v1.0.0 sampled 100,000 typed triples',
         'seeds':list(SEEDS), 'feature_modes':{},'notes':[
         'Five seeds vary train/validation/test split and initial parameters on the SAME sampled graph.',
         'The candidate pools differ across split seeds and also from the earlier 10k experiment.',
         'No claim of independent random graph samples or statistical significance.',
         'One sampled fixed-size graph, not a controlled training-size learning curve.',
         '25 epochs with max 4096 positive ranking triples per epoch; all train links are nevertheless used by graph message passing.',
         'No Voyager HNSW or LLM in this score comparison.']}
    rows=[]
    for feature in FEATURES:
        reports=[json.loads((root/feature/f'evaluation_seed{seed}.json').read_text()) for seed in SEEDS]
        if not all(report['query_count']==128 for report in reports):raise ValueError('Expected 128 evaluation queries in every seed')
        summary={}
        for method in METHODS:
            stats={}
            for metric in METRICS:
                xs=np.array([float(d['methods'][method]['test'][metric]) for d in reports])
                stats[metric]={'mean':float(xs.mean()),'std_sample':float(xs.std(ddof=1)),
                               'by_seed':{str(s):float(x) for s,x in zip(SEEDS,xs)}}
                rows.append({'feature':feature,'method':method,'metric':metric,
                             'mean':float(xs.mean()),'std_sample':float(xs.std(ddof=1))})
            summary[method]=stats
        delta_gnn=np.array([d['methods']['gnn_only']['test']['mrr@10']-d['methods']['bfs_only']['test']['mrr@10'] for d in reports])
        delta_fuse=np.array([d['methods']['bfs_gnn']['test']['mrr@10']-d['methods']['bfs_only']['test']['mrr@10'] for d in reports])
        delta_fuse_gnn=np.array([d['methods']['bfs_gnn']['test']['mrr@10']-d['methods']['gnn_only']['test']['mrr@10'] for d in reports])
        out['feature_modes'][feature]={
            'methods':summary,'alpha':{str(seed):float(d['calibration']['alpha']) for seed,d in zip(SEEDS,reports)},
            'paired_mrr_differences':{
                'gnn_minus_bfs':{'mean':float(delta_gnn.mean()),'wins':int(sum(delta_gnn>1e-12)),'by_seed':delta_gnn.tolist()},
                'fused_minus_bfs':{'mean':float(delta_fuse.mean()),'wins':int(sum(delta_fuse>1e-12)),'by_seed':delta_fuse.tolist()},
                'fused_minus_gnn':{'mean':float(delta_fuse_gnn.mean()),'wins':int(sum(delta_fuse_gnn>1e-12)),'by_seed':delta_fuse_gnn.tolist()},
            },
            'candidate_counts_test':{str(seed):d['candidate_counts_test'] for seed,d in zip(SEEDS,reports)},
            'train_test_counts':{str(seed):{key:d[key] for key in ('train_triples','validation_triples','test_triples','query_count')}
                                 for seed,d in zip(SEEDS,reports)},
        }
    (root/'aggregate_5seed.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    with (root/'aggregate_5seed.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=['feature','method','metric','mean','std_sample']);w.writeheader();w.writerows(rows)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results'))
    a=p.parse_args();r=build(a.results)
    for feature in FEATURES:
        print(feature)
        for m in METHODS:
            x=r['feature_modes'][feature]['methods'][m]
            print(m,'mrr@10',round(x['mrr@10']['mean'],6),'hits@10',round(x['hits@10']['mean'],6))
