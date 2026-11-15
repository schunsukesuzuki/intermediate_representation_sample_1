"""Computationally optimized, semantically equivalent v6 triple ranking benchmark.

Three methods use IDENTICAL eligible target IDs. GraphSAGE embedding scores, BFS
shortest paths and validation-tuned percentile-rank fusion remain unchanged.
No test targets are used to select alpha. No approximate nearest-neighbor ranking.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from hashlib import sha256
import numpy as np
from gnn_rag.ontology import RELATIONS

KS=(1,3,5,10)


def grouped_targets(graph, triples):
    out=defaultdict(set)
    for h,r,t in np.asarray(triples,dtype=np.int32):
        h,r,t=int(h),int(r),int(t)
        out[h,r].add(t)
        spec=RELATIONS.get(graph.relation_names[r])
        if spec is not None and not spec.directed and spec.source_kind==spec.target_kind:
            out[t,r].add(h)
    return out


def choose_queries(graph,labels,partition,seed,max_queries):
    if max_queries is None:
        return sorted(labels)
    if max_queries < 1:
        raise ValueError('max_queries must be positive')
    by_rel=defaultdict(list)
    for h,r in sorted(labels):
        digest=sha256(f'{seed}|{partition}|{graph.ids[h]}|{graph.relation_names[r]}'.encode()).digest()
        by_rel[r].append((digest,(h,r)))
    for r in by_rel: by_rel[r].sort()
    rels=sorted(by_rel); keys=[]
    while len(keys)<max_queries and rels:
        active=[]
        for r in rels:
            if by_rel[r] and len(keys)<max_queries: keys.append(by_rel[r].pop(0)[1])
            if by_rel[r]: active.append(r)
        rels=active
    return keys


def prepare_cache(graph,train):
    block=defaultdict(set)
    adjacency=[set() for _ in range(graph.n_nodes)]
    kinds=defaultdict(list)
    for i,kind in enumerate(graph.node_kinds): kinds[kind].append(i)
    for h,r,t in train:
        h,r,t=int(h),int(r),int(t)
        block[h,r].add(t)
        spec=RELATIONS.get(graph.relation_names[r])
        if spec is not None and not spec.directed and spec.source_kind==spec.target_kind:
            block[t,r].add(h)
        adjacency[h].add(t);adjacency[t].add(h)
    bykind={name:np.asarray(inds,dtype=np.int32) for name,inds in kinds.items()}
    allnodes=np.arange(graph.n_nodes,dtype=np.int32)
    degree=np.asarray([len(v) for v in adjacency],dtype=np.int32)
    return block,adjacency,bykind,allnodes,degree


def bfs_single(head,adj):
    d={head:0};queue=deque([head])
    while queue:
        u=queue.popleft()
        for v in adj[u]:
            if v not in d:
                d[v]=d[u]+1;queue.append(v)
    return d

@dataclass
class QR:
    head: int
    relation: int
    candidates: np.ndarray
    positives: frozenset
    gnn: np.ndarray
    bfs: np.ndarray


def make_query_ranks_fast(graph,split,ranker,partition,seed,max_queries):
    labels=grouped_targets(graph,getattr(split,partition))
    val=grouped_targets(graph,split.validation) if partition=='test' else {}
    block,adj,bykind,allnodes,degree=prepare_cache(graph,split.train)
    keys=choose_queries(graph,labels,partition,seed,max_queries)
    distance_cache={};result=[]
    for h,r in keys:
        spec=RELATIONS.get(graph.relation_names[r]); kind=spec.target_kind if spec else None
        base=allnodes if kind is None else bykind.get(kind,np.empty(0,dtype=np.int32))
        blocked=block[h,r]; valblocked=val.get((h,r),set())
        candidates=np.asarray([int(i) for i in base if int(i)!=h and int(i) not in blocked and int(i) not in valblocked],dtype=np.int32)
        positive=labels[h,r]
        if not positive.issubset(set(candidates.tolist())): raise ValueError('positive not in candidates')
        if not len(candidates):continue
        gnn=np.asarray(ranker.exact(h,r,candidates),dtype=np.int32)
        if h not in distance_cache:distance_cache[h]=bfs_single(h,adj)
        d=distance_cache[h]
        bfsscore=np.asarray([d.get(int(i),float('inf')) for i in candidates],dtype=float)
        order=np.lexsort((candidates,-degree[candidates],bfsscore))
        bfs=candidates[order]
        if set(gnn)!=set(candidates) or set(bfs)!=set(candidates):raise ValueError('invalid rankings')
        result.append(QR(h,r,candidates,frozenset(positive),gnn,bfs))
    return result


def fused(q,alpha):
    if alpha==0.0:return q.bfs
    if alpha==1.0:return q.gnn
    n=len(q.candidates)
    # Both q.gnn and q.bfs contain same candidate ids; candidate arrays are in
    # increasing node-id order, so positions are indexed via searchsorted.
    gidx=np.searchsorted(q.candidates,q.gnn)
    bidx=np.searchsorted(q.candidates,q.bfs)
    gp=np.empty(n,dtype=np.int32);bp=np.empty(n,dtype=np.int32)
    gp[gidx]=np.arange(n,dtype=np.int32)
    bp[bidx]=np.arange(n,dtype=np.int32)
    scale=max(n-1,1)
    # Scalar-Python floating point expression is reproduced in numpy float64.
    score=alpha*(1.0-gp/scale)+(1.0-alpha)*(1.0-bp/scale)
    return q.candidates[np.lexsort((q.candidates,gp,bp,-score))]


def measure(order,positives,ks=KS):
    positive=set(positives)
    pos=np.where(np.isin(order[:max(ks)],list(positive)))[0]+1
    return {name:float(value) for k in ks for name,value in (
            (f'hits@{k}',bool(np.any(pos<=k))),
            (f'recall@{k}',np.count_nonzero(pos<=k)/len(positive)),
            (f'mrr@{k}',1.0/int(pos[0]) if len(pos) and pos[0]<=k else 0.0))}


def agg(queries,alpha,ks=KS):
    metrics=defaultdict(list)
    for q in queries:
        vals=measure(fused(q,alpha),q.positives,ks)
        for k,v in vals.items():metrics[k].append(v)
    return {k:float(np.mean(vals)) for k,vals in metrics.items()}


def compare_three_fast(graph,split,ranker,seed,max_queries=128,ks=KS,tune_k=10):
    val=make_query_ranks_fast(graph,split,ranker,'validation',seed,max_queries)
    test=make_query_ranks_fast(graph,split,ranker,'test',seed,max_queries)
    if not val or not test:raise ValueError('empty eval')
    grid=[]
    for i in range(21):
        a=i/20;m=agg(val,a,tuple(sorted({1,tune_k})))
        grid.append({'alpha':a,'validation_mrr':m[f'mrr@{tune_k}'],'validation_hits1':m['hits@1']})
    chosen=max(grid,key=lambda x:(x['validation_mrr'],x['validation_hits1'],-x['alpha']))
    alpha=chosen['alpha']; methods={'gnn_only':1.0,'bfs_only':0.0,'bfs_gnn':alpha}
    rows=[];report_methods={}
    for name,a in methods.items():
        score_test=agg(test,a,ks);score_val=agg(val,a,ks)
        report_methods[name]={'test':score_test,'validation':score_val}
    # Group per-query diagnostics by query, then by method, as in the
    # reference evaluator. The ranking/metrics/calibration remain unchanged.
    for q in test:
        for name,a in methods.items():
            rows.append({'method':name,'source':graph.ids[q.head],
                         'relation':graph.relation_names[q.relation],
                         'n_candidates':len(q.candidates), 'n_relevant':len(q.positives),
                         'alpha':a, **measure(fused(q,a),q.positives,ks)})
    report={'task':'held_out_relation_typed_triple_retrieval',
            'dataset':'set by caller', 'model':'GraphSAGE + relation query',
            'comparison':'GNN only vs BFS only vs validation-tuned BFS+GNN',
            'methods':report_methods,
            'calibration':{'alpha':alpha,'objective':f'validation_mrr@{tune_k}','grid':grid,
                           'validation_queries':len(val),'uses_test_labels':False,
                           'fallback_to_bfs':alpha==0.0,
                           'chosen_validation_mrr':chosen['validation_mrr']},
            'query_count':len(test),'evaluation_query_cap_per_split':max_queries,
            'validation_query_count':len(val),
            'candidate_counts_test':{'min':min(len(q.candidates) for q in test),
                        'median':float(np.median([len(q.candidates) for q in test])),
                        'max':max(len(q.candidates) for q in test)},
            'test_triples':len(split.test),'validation_triples':len(split.validation),
            'train_triples':len(split.train),'ks':list(ks),'seed':seed,
            'fusion_formula':'alpha*GNN_rank_percentile + (1-alpha)*BFS_rank_percentile',
            'candidate_rule':'same relation-compatible tails excluding train links; validation positives also filtered on test',
            'bfs_graph':'train triples only, undirected adjacency; distance then degree then ID',
            'gnn_retrieval':'exact all-candidate relation-conditioned score (NOT HNSW)',
            'notes':['All rankings and alpha selection obey the original v6 method.',
                     'This is a transductive held-out link task, not clinical validation.',
                     'Sampling from full graph precedes train/test split.',
                     'Only validation was used to select alpha; no test tuning.',
                     'No artificial cluster-label feature added.',
                     'Reduced Python overhead for 100k triples; semantically equivalent to v6.']}
    return report,rows
