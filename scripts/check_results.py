import csv, json
from pathlib import Path
p=Path('/mnt/data/gnn_hetionet_real_run')
nodes=list(csv.DictReader((p/'prepared/nodes.csv').open()))
edges=list(csv.DictReader((p/'prepared/edges.csv').open()))
assert len(nodes)==495 and len(edges)==10000
keys={(e['source'],e['relation'],e['target']) for e in edges}
assert len(keys)==len(edges)
ids={x['node_id'] for x in nodes}
assert all(e['source'] in ids and e['target'] in ids for e in edges)
for d in (p/'output',p/'kind_only'):
 for seed in range(42,47):
  j=json.loads((d/f'evaluation_seed{seed}.json').read_text())
  assert j['training_metrics']['epochs']==25
  assert j['training_metrics']['feature_mode']==('kind_only' if d.name=='kind_only' else 'kind_name')
  assert j['query_count']==128 and j['validation_query_count']==128
  assert j['calibration']['uses_test_labels'] is False
  assert 0<=j['calibration']['alpha']<=1
  assert sum([j['train_triples'],j['validation_triples'],j['test_triples']])==10000
  with (d/f'evaluation_per_query_seed{seed}.csv').open() as f:
   rows=list(csv.DictReader(f))
  assert len(rows)==3*128
  assert {x['method'] for x in rows}=={'gnn_only','bfs_only','bfs_gnn'}
  ref={(x['source'],x['relation']):(x['n_candidates'],x['n_relevant']) for x in rows if x['method']=='gnn_only'}
  assert len(ref)==128
  for row in rows:
   assert (row['n_candidates'],row['n_relevant'])==ref[(row['source'],row['relation'])]
   for met in ['mrr@10','hits@10','recall@10']:
    assert 0<=float(row[met])<=1
  for method in ['gnn_only','bfs_only','bfs_gnn']:
   m=sum(float(x['mrr@10']) for x in rows if x['method']==method)/128
   assert abs(m-j['methods'][method]['test']['mrr@10']) < 1e-7
print('VERIFIED 10k unique triples, node integrity, 10 full training/evaluations, 128 matched queries per split, 384 rows per run, independent alpha tuning')
