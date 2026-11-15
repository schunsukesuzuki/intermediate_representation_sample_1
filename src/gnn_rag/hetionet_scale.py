"""Resource-bounded sampler from *full* Hetionet v1.0 TSV + SIF.gz.

Selects a structural, multiseed induced subgraph, then randomly retains a
spanning forest plus additional triples up to target_triples. It does NOT
manufacture node names or cluster labels. Sampling runs before a subsequent
train/validation/test split and is NOT a random IID sample of Hetionet.

Full SIF is streamed into compact int32 arrays, not a list of Python string
triples; the raw file remains external and is never bundled in the ZIP.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def load_full_hetionet(nodes_tsv: Path, edges_sif_gz: Path):
    nodes = {}
    with nodes_tsv.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        if not {'id', 'name', 'kind'}.issubset(set(reader.fieldnames or [])):
            raise ValueError('Node TSV must have id, name, kind columns')
        for row in reader:
            if row['id']:
                nodes[row['id']] = (row['name'], row['kind'])
    ids = tuple(sorted(nodes))
    lookup = {name: i for i, name in enumerate(ids)}
    rel_lookup = {}
    src, dst, rel = array('i'), array('i'), array('i')
    with gzip.open(edges_sif_gz, 'rt', encoding='utf-8-sig', newline='') as fh:
        reader = csv.reader(fh, delimiter='\t')
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            if len(row) != 3:
                raise ValueError('SIF requires source, metaedge, target')
            source, relation, target = (x.strip() for x in row)
            if (source.lower(), relation.lower(), target.lower()) == ('source', 'metaedge', 'target'):
                continue
            if source not in lookup or target not in lookup or not relation:
                raise ValueError('SIF row refers to missing node or blank relation')
            if source == target:
                continue
            rel_id = rel_lookup.setdefault(relation, len(rel_lookup))
            src.append(lookup[source]); dst.append(lookup[target]); rel.append(rel_id)
    if len(src) < 100:
        raise ValueError('Hetionet source contains too few edges')
    return ids, nodes, tuple(rel_lookup), (np.frombuffer(src, dtype=np.int32).copy(),
                                          np.frombuffer(dst, dtype=np.int32).copy(),
                                          np.frombuffer(rel, dtype=np.int32).copy())


def sample_indices(ids: tuple[str, ...], kinds: list[str], src: np.ndarray,
                   dst: np.ndarray, rel: np.ndarray | None = None, *, target_triples: int, max_nodes: int,
                   seed: int, focus_index: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Randomized multi-start frontier growth, capped by node and edge budgets.

    Returns indices of selected nodes and selected *source* triple rows. A
    randomized spanning forest is always kept before adding other edges.
    """
    if target_triples < 100 or max_nodes < 50:
        raise ValueError('Use target_triples>=100 and max_nodes>=50')
    if max_nodes > len(ids):
        max_nodes = len(ids)
    rng = np.random.default_rng(seed)
    n = len(ids)
    degree = np.bincount(np.concatenate([src, dst]), minlength=n)
    from_idx = np.concatenate([src, dst])
    to_idx = np.concatenate([dst, src])
    order = np.argsort(from_idx, kind='stable')
    neighbors = to_idx[order]
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(from_idx, minlength=n))
    # Prevent a single disease hub from defining the whole dataset. The
    # extra seeds are stratified by biological *node type*, not latent cluster.
    seeds = []
    if focus_index is not None:
        seeds.append(int(focus_index))
    for kind in sorted(set(kinds)):
        available = [j for j, k in enumerate(kinds) if k == kind and degree[j] >= 2 and j not in seeds]
        if available:
            seeds.append(int(rng.choice(available)))
    selected = np.zeros(n, dtype=bool)
    chosen = []
    frontier = []
    induced = 0
    def add(node: int):
        nonlocal induced
        if selected[node]:
            return
        nb = neighbors[offsets[node]:offsets[node + 1]]
        induced += int(np.count_nonzero(selected[nb]))
        selected[node] = True
        chosen.append(node)
        frontier.append(node)
    for node in seeds[:max_nodes]:
        add(node)
    # Grows until enough induced edges exist; choose a frontier source and a
    # not-yet-selected neighbor, never examine validation/test labels.
    while len(chosen) < max_nodes and induced < int(target_triples * 1.25):
        if not frontier:
            rest = np.flatnonzero(~selected & (degree > 0))
            if not len(rest):
                break
            add(int(rng.choice(rest)))
            continue
        k = int(rng.integers(len(frontier)))
        u = frontier[k]
        nb = neighbors[offsets[u]:offsets[u+1]]
        free = nb[~selected[nb]]
        if not len(free):
            frontier[k] = frontier[-1]
            frontier.pop()
            continue
        add(int(rng.choice(free)))
    selected_rows = np.flatnonzero(selected[src] & selected[dst])
    # Sample DISTINCT typed triples where a relation array is provided.
    # The synthetic sampler unit tests also use this option to exercise this.
    if rel is not None and len(selected_rows):
        values = np.stack((src[selected_rows], rel[selected_rows], dst[selected_rows]), axis=1)
        _, first = np.unique(values, axis=0, return_index=True)
        selected_rows = selected_rows[np.sort(first)]
    if len(selected_rows) < target_triples:
        raise ValueError(
            f'Only {len(selected_rows)} induced triples with {len(chosen)} nodes; '
            f'raise HETIONET_MAX_NODES (currently {max_nodes}) or reduce '
            f'HETIONET_TARGET_TRIPLES (currently {target_triples}).'
        )
    # Keep a randomized spanning forest so all selected nodes used in training
    # remain linked where possible, then uniform-fill from the remaining
    # induced triples. Uses ALL graph rows only for data sampling, BEFORE split.
    p = np.arange(n, dtype=np.int32)
    def find(a):
        while p[a] != a:
            p[a] = p[p[a]]
            a = int(p[a])
        return a
    forest = []
    perm = rng.permutation(selected_rows)
    for ix in perm:
        a, b = find(int(src[ix])), find(int(dst[ix]))
        if a != b:
            p[a] = b
            forest.append(int(ix))
    if len(forest) > target_triples:
        raise ValueError('Forest exceeds triple budget')
    chosen_rows = set(forest)
    required = target_triples - len(chosen_rows)
    if required:
        other = np.asarray([int(ix) for ix in perm if int(ix) not in chosen_rows], dtype=np.int64)
        chosen_rows.update(int(i) for i in rng.choice(other, size=required, replace=False))
    final_rows = np.asarray(sorted(chosen_rows), dtype=np.int64)
    used = np.unique(np.concatenate([src[final_rows], dst[final_rows]])).astype(np.int32)
    return used, final_rows, {
        'selected_nodes_before_final_edge_sampling': len(chosen),
        'induced_triples_before_final_edge_sampling': len(selected_rows),
        'spanning_forest_triples': len(forest),
        'full_source_triple_count': len(src),
        'full_source_node_count': n,
        'sampling': 'node-type-stratified multi-seed frontier + induced edges + spanning forest + uniform remainder',
        'selection_based_on_full_unsplit_graph': True,
    }


def convert_full_hetionet(nodes_tsv: Path, edges_sif_gz: Path, output: Path,
                          *, target_triples: int = 10_000, max_nodes: int = 5_000,
                          seed: int = 42, focus: str = "Alzheimer's disease") -> dict:
    if not nodes_tsv.is_file() or not edges_sif_gz.is_file():
        raise FileNotFoundError('Missing full Hetionet raw TSV / SIF.gz inputs')
    ids, attrs, relations, (src, dst, rel) = load_full_hetionet(nodes_tsv, edges_sif_gz)
    names = [attrs[x][0] for x in ids]
    focus_matches = [i for i, name in enumerate(names) if name.casefold() == focus.casefold()]
    focus_id = ids[focus_matches[0]] if len(focus_matches) == 1 else None
    kinds = [attrs[x][1] for x in ids]
    used, edge_rows, stats = sample_indices(ids, kinds, src, dst, rel,
        target_triples=target_triples, max_nodes=max_nodes, seed=seed,
        focus_index=focus_matches[0] if len(focus_matches) == 1 else None)
    triples = sorted({(ids[int(src[i])], relations[int(rel[i])], ids[int(dst[i])])
                      for i in edge_rows})
    if len(triples) < int(.99 * target_triples):
        raise ValueError('Too many duplicate source triples; increase the requested selection budget')
    used_keys = sorted({key for h, _, t in triples for key in (h, t)})
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'nodes.csv.tmp').open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['node_id','name','kind','topic','description'])
        writer.writeheader()
        for key in used_keys:
            name, kind = attrs[key]
            writer.writerow({'node_id': key, 'name': name, 'kind': kind, 'topic': kind,
                             'description': f'Hetionet node category: {kind}; no narrative provided by TSV.'})
    with (output / 'edges.csv.tmp').open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['source','relation','target','weight'])
        writer.writeheader()
        for h, r, t in triples:
            writer.writerow({'source': h, 'relation': r, 'target': t, 'weight': 1.0})
    (output / 'nodes.csv.tmp').replace(output / 'nodes.csv')
    (output / 'edges.csv.tmp').replace(output / 'edges.csv')
    meta = {
        'schema_version': 6, 'triple_preserving': True,
        'sampling_mode': 'full_hetionet_structural_sample',
        'focus_id': focus_id if focus_id in used_keys else used_keys[0],
        'requested_focus': focus,
        'focus_in_sample': focus_id in used_keys,
        'nodes': len(used_keys), 'edges': len(triples),
        'requested_target_triples': target_triples, 'requested_max_nodes': max_nodes,
        'seed': seed, 'types': sorted({attrs[key][1] for key in used_keys}),
        'relation_counts': dict(sorted(Counter(r for _,r,_ in triples).items())),
        'source': 'hetio/hetionet v1.0 TSV + SIF.gz (full original data)',
        'caveats': [
            'Dataset uses an induced, multi-source structural SAMPLE, not an IID draw.',
            'Sampling precedes holdout split; TRAIN adjacency excludes held-out links.',
            'Raw SIF lacks per-edge provenance and licensing annotations.',
            'This dataset is only a held-out link ranking test, not clinical validation.',
        ],
        **stats,
    }
    (output / 'dataset_meta.json.tmp').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output / 'dataset_meta.json.tmp').replace(output / 'dataset_meta.json')
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nodes', type=Path, required=True)
    p.add_argument('--edges', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('/app/data/hetionet_demo'))
    p.add_argument('--target-triples', type=int, default=10000)
    p.add_argument('--max-nodes', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--focus', default="Alzheimer's disease")
    a = p.parse_args()
    print(json.dumps(convert_full_hetionet(a.nodes,a.edges,a.output,target_triples=a.target_triples,
          max_nodes=a.max_nodes,seed=a.seed,focus=a.focus), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
