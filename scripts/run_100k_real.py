#!/usr/bin/env python3
"""Run v6 GraphSAGE/typed-query/fusion code on attached Hetionet 100,000 Triple sample.
This local execution adapter replaces only the Polars CSV boundary with pandas
because Polars and Voyager are absent in this execution environment.
The three ranking methods all use exact candidate search (as v6 does).
"""
import argparse, csv, hashlib, json, os, sys, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from gnn_rag.triple_learning import triple_split, train_triples
from gnn_rag.typed_search import RelationRanker
from gnn_rag.fusion_evaluation import markdown_comparison
from gnn_rag.fast_evaluation import compare_three_fast

class ArrayColumn:
    def __init__(self, values): self.values=list(values)
    def to_list(self):return self.values

class PandasNodeBoundary:
    def __init__(self, df): self.df=df
    def get_column(self, name):return ArrayColumn(self.df[name].tolist())

@dataclass(frozen=True)
class MinimalGraphData:
    nodes: object
    edges: object
    features: np.ndarray
    senders: np.ndarray
    receivers: np.ndarray
    weights: np.ndarray
    ids: tuple[str,...]
    id_to_index: dict[str,int]
    relation_ids: np.ndarray
    relation_names: tuple[str,...]
    node_kinds: tuple[str,...]
    @property
    def n_nodes(self):return len(self.ids)

def load_csv_no_polars(nodes_csv, edges_csv):
    nodes=pd.read_csv(nodes_csv, dtype=str).sort_values('node_id').reset_index(drop=True)
    edges=pd.read_csv(edges_csv, dtype=str).sort_values(['source','target'],kind='stable').reset_index(drop=True)
    assert nodes.node_id.is_unique
    ids=tuple(nodes.node_id.tolist()); id_to_index={x:i for i,x in enumerate(ids)}
    r_names=tuple(sorted(set(edges.relation.tolist())))
    r_map={r:i for i,r in enumerate(r_names)}
    sources=np.array([id_to_index[x] for x in edges.source],dtype=np.int32)
    targets=np.array([id_to_index[x] for x in edges.target],dtype=np.int32)
    rel_ids=np.array([r_map[x] for x in edges.relation],dtype=np.int32)
    assert len({(int(s),int(r),int(t)) for s,r,t in zip(sources,rel_ids,targets)})==len(edges)
    kinds=tuple(nodes.kind.tolist())
    topics=sorted(nodes.topic.unique().tolist()); types=sorted(set(kinds))
    feats=np.zeros((len(ids),len(topics)+len(types)),dtype=np.float32)
    tm={x:i for i,x in enumerate(topics)}; km={x:i+len(topics) for i,x in enumerate(types)}
    for i,(topic,kind) in enumerate(zip(nodes.topic,kinds)):
        feats[i,tm[topic]]=1.;feats[i,km[kind]]=1.
    return MinimalGraphData(PandasNodeBoundary(nodes), edges, feats, sources, targets,
               edges.weight.astype(float).to_numpy(dtype=np.float32), ids, id_to_index,
               rel_ids,r_names,kinds)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path('data/hetionet_demo'))
    ap.add_argument('--output',type=Path,default=Path('results/kind_name'))
    ap.add_argument('--epochs',type=int,default=25)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--eval-max-queries',type=int,default=128)
    ap.add_argument('--train-max-positives',type=int,default=4096)
    ap.add_argument('--feature-mode',choices=['kind_name','kind_only'],default='kind_name')
    ap.add_argument('--source-archive', type=Path, default=None, help='Optional local source ZIP for checksum; not redistributed.')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    t=time.time()
    g=load_csv_no_polars(a.input/'nodes.csv', a.input/'edges.csv')
    print('GRAPH',g.n_nodes,'triples',len(g.senders),'relations',len(g.relation_names),flush=True)
    split=triple_split(g,seed=a.seed)
    print('SPLIT train=',len(split.train),'validation=',len(split.validation),'test=',len(split.test),flush=True)
    train_keys={(min(int(h),int(t)),max(int(h),int(t))) for h,r,t in split.train}
    val_keys={(min(int(h),int(t)),max(int(h),int(t))) for h,r,t in split.validation}
    test_keys={(min(int(h),int(t)),max(int(h),int(t))) for h,r,t in split.test}
    assert not(train_keys & val_keys or train_keys & test_keys or val_keys & test_keys)
    print('LEAKAGE CHECK pair-disjoint PASS',flush=True)
    trained=train_triples(g,split,seed=a.seed,epochs=a.epochs,
             max_positives_per_epoch=a.train_max_positives,feature_mode=a.feature_mode)
    print('TRAIN COMPLETE',trained.metrics,'seconds',time.time()-t,flush=True)
    rv=np.asarray(trained.params['relation_vectors'],dtype=np.float32)
    ranker=RelationRanker(g.ids,g.node_kinds,g.relation_names,trained.embeddings,rv)
    report,rows=compare_three_fast(g,split,ranker,seed=a.seed,max_queries=a.eval_max_queries)
    report['full_source']='uploaded hetionet-v1.0.0.zip, official TSV and SIF.gz'
    if a.source_archive is not None:
        report['source_archive_sha256']=hashlib.sha256(a.source_archive.read_bytes()).hexdigest()
    report['node_count']=g.n_nodes
    report['input_triples']=len(g.senders)
    report['seed']=a.seed
    report['training_metrics']=trained.metrics
    report['timing_seconds']=round(time.time()-t,2)
    report['execution_notes']=['Real Hetionet TSV / SIF source, v6 sampler 100000 triples.',
         'v6 exact GraphSAGE and rank fusion. A pandas-backed CSV boundary substitutes for unavailable Polars only.',
         'No Voyager used in these three strict ranking baselines; GNN exhaustive exact top-K.',
         'Evaluation loops optimized using precomputed training adjacency/blocked candidates and NumPy lexsort; metrics verified bitwise-equivalent to original v6 on 10k seed42.',
         'Sampling uses full unsplit graph before train/validation/test. This is a transductive sampled-link task, not a strictly train-only sampling process.',
         'Same-node-type negatives are not necessarily true negative biomedical links.',
         'Feature_mode '+a.feature_mode+' (actual KG natural names, no synthetic clan features).']
    (a.output/f'evaluation_seed{a.seed}.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (a.output/f'evaluation_seed{a.seed}.md').write_text(markdown_comparison(report),encoding='utf-8')
    with (a.output/f'evaluation_per_query_seed{a.seed}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(a.output/f'gnn_embeddings_seed{a.seed}.npz',ids=np.array(g.ids),embeddings=trained.embeddings,
                         relation_names=np.array(g.relation_names),relation_vectors=rv)
    print('SCORES',flush=True)
    for method,entry in report['methods'].items():
        print(method, {k:round(entry['test'][k],6) for k in ['hits@1','hits@10','mrr@10','recall@10']},flush=True)
    print('ALPHA',report['calibration']['alpha'],'seconds',time.time()-t,flush=True)

if __name__=='__main__':main()
