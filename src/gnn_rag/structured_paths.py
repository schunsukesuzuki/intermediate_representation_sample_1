"""v13: deterministic, provenance-preserving intermediate representation.

Only triples appearing in the typed, verified TRAIN-edge catalog may enter a
structured path. Traversal direction is explicitly distinct from source -> target.
LLM-generated text is never accepted as a registered edge or structured path.
"""
from __future__ import annotations

from typing import Any


def build_verified_paths(catalog: list[dict], evidence_paths: list[dict],
                         selected_ids: list[str], *, max_paths: int = 5,
                         max_hops: int = 4) -> list[dict[str, Any]]:
    facts = {
        (str(f.get('source')), str(f.get('relation')), str(f.get('target'))): f
        for f in catalog
        if all(f.get(k) for k in ('source', 'relation', 'target', 'evidence_id'))
    }
    selected = set(selected_ids)
    output: list[dict[str, Any]] = []
    for raw_path in evidence_paths[:max_paths]:
        nodes = raw_path.get('nodes', [])
        raw_edges = raw_path.get('edges', [])
        if not isinstance(nodes, (tuple, list)) or not isinstance(raw_edges, (tuple, list)):
            continue
        steps: list[dict[str, Any]] = []
        for i, raw_edge in enumerate(raw_edges[:max_hops]):
            if i + 1 >= len(nodes) or not isinstance(raw_edge, dict):
                break
            src, rel, dst = (str(raw_edge.get(k, '')) for k in ('source', 'relation', 'target'))
            fact = facts.get((src, rel, dst))
            if fact is None:
                break   # Do not skip a missing edge and invent a longer contiguous path.
            curr, nxt = str(nodes[i]), str(nodes[i+1])
            if (curr, nxt) == (src, dst):
                traversed = 'forward'
            elif (curr, nxt) == (dst, src):
                traversed = 'reverse'
            else:
                break   # Edge exists but cannot connect this path's adjacent nodes.
            steps.append({
                'hop': i + 1, 'evidence_id': str(fact['evidence_id']),
                'source_id': src, 'source_name': str(fact.get('source_name', src)),
                'source_kind': str(fact.get('source_kind', '')),
                'relation_code': rel, 'relation_label': str(fact.get('label', rel)),
                'provenance': str(fact.get('provenance', '')),
                'registered_statement': str(fact.get('statement', ''))[:450],
                'target_id': dst, 'target_name': str(fact.get('target_name', dst)),
                'target_kind': str(fact.get('target_kind', '')),
                'traversal_from': curr, 'traversal_to': nxt,
                'traversal_direction': traversed,
                'selected_for_answer': str(fact['evidence_id']) in selected,
                'observed_train_edge': True,
            })
        if not steps:
            continue
        output.append({
            'path_id': f'P{len(output)+1}',
            'traversal_node_ids': [str(n) for n in nodes[:len(steps)+1]],
            'steps': steps,
            'status': 'verified_observed_train_path',
        })
    return output


def format_verified_paths(paths: list[dict]) -> str:
    """Readable, deterministic path display; not a Qwen explanation."""
    if not paths:
        return 'この方式について型・連続性を検証できたTrainパスはありません。'
    lines: list[str] = []
    for path in paths:
        lines.append(f"{path['path_id']}（観測済みTrainエッジ）")
        for step in path['steps']:
            arrow = '順方向' if step['traversal_direction'] == 'forward' else '逆方向に探索'
            selected = '採用' if step['selected_for_answer'] else '未採用（エッジは存在）'
            from_name = (step['source_name'] if step['traversal_direction']=='forward'
                         else step['target_name'])
            to_name = (step['target_name'] if step['traversal_direction']=='forward'
                       else step['source_name'])
            lines.append(
                f"  hop {step['hop']} [{step['evidence_id']}; {selected}] "
                f"{step['source_name']}（{step['source_kind']}） "
                f"--[{step['relation_code']}: {step['relation_label']}]--> "
                f"{step['target_name']}（{step['target_kind']}） "
                f"[探索順: {from_name}→{to_name}（{arrow}）; 原登録の向きはsource→target]"
            )
            if step['provenance']:
                lines.append(f"    関係定義・出典: {step['provenance']}")
    return '\n'.join(lines)


def build_method_audit(results: dict[str, dict]) -> dict[str, dict]:
    """Exact selected KG facts by method; Plain Qwen must have zero KG facts."""
    audit: dict[str, dict] = {}
    for key in ('plain_qwen', 'bfs_only', 'gnn_only', 'bfs_gnn'):
        entry = results.get(key, {})
        if key == 'plain_qwen':
            audit[key] = {'source': 'no_kg_retrieval', 'selected_train_edges': [],
                          'path_count': 0, 'note': 'Plain Qwen has no supplied KG evidence.'}
            continue
        selected = set(entry.get('selected_evidence_ids', []))
        facts = [
            {k: f.get(k, '') for k in (
                'evidence_id', 'source', 'source_name', 'source_kind', 'relation',
                'label', 'provenance', 'target', 'target_name', 'target_kind'
            )}
            for f in entry.get('catalog', []) if f.get('evidence_id') in selected
        ]
        audit[key] = {'source': 'observed_train_kg', 'selected_train_edges': facts,
                      'path_count': len(entry.get('structured_paths', [])),
                      'note': 'Relations are independently typed; selected edges are not all identical.'}
    return audit


def build_cross_factual_comparison(audit: dict[str, dict]) -> dict[str, object]:
    """Deterministic triple comparison; IDs are local to each method's catalog.

    Only (source, relation, target) is a cross-method identity. 'E5' in BFS
    is NOT necessarily the same triple as 'E5' in GNN.
    """
    keys = ('bfs_only', 'gnn_only', 'bfs_gnn')
    triples = {}
    details = {}
    for method in keys:
        facts = audit.get(method, {}).get('selected_train_edges', [])
        indexed = {
            (str(f.get('source')), str(f.get('relation')), str(f.get('target'))): f
            for f in facts if all(f.get(k) for k in ('source', 'relation', 'target'))
        }
        triples[method] = set(indexed)
        details[method] = indexed
    common = set.intersection(*(triples[k] for k in keys))
    unique = {k: triples[k] - set.union(*(triples[j] for j in keys if j != k))
              for k in keys}
    def readable(triple, facts):
        f = facts[triple]
        return (str(f.get('source_name') or triple[0]) + ' --[' + triple[1] + ': ' +
                str(f.get('label') or triple[1]) + ']--> ' +
                str(f.get('target_name') or triple[2]))
    # Headline source separation and exact edges precede ALL LLM commentary.
    lines = ['Plain Qwen：KG検索なし。登録Trainエッジ数は0（自由生成）。']
    names = {'bfs_only':'BFS only','gnn_only':'GNN only','bfs_gnn':'BFS+GNN'}
    for key in keys:
        lines.append(f'{names[key]}：採用済み観測Trainトリプル {len(triples[key])} 件。')
        for triple in sorted(triples[key]):
            lines.append('  - ' + readable(triple, details[key]))
    lines.append(f'KG方式3種すべてに共通する登録Trainトリプル: {len(common)} 件。')
    for triple in sorted(common):
        lines.append('  - ' + readable(triple, details['bfs_only']))
    for key in keys:
        lines.append(f'{names[key]}のみに現れる採用トリプル: {len(unique[key])} 件。')
        for triple in sorted(unique[key]):
            lines.append('  - ' + readable(triple, details[key]))
    lines.append('注：比較は(source, relation, target)をキーに行い、方法ごとに付け直されるE1等の局所IDを比較キーに使いません。')
    lines.append('これらは観測済みTrain上の選択結果であり、未採用=非存在でも、医学的因果・正確性の証明でもありません。')
    return {
        'plain_qwen_kg_edges': 0,
        'selected_counts': {k:len(triples[k]) for k in keys},
        'common_triples': [list(t) for t in sorted(common)],
        'unique_triples': {k:[list(t) for t in sorted(unique[k])] for k in keys},
        'summary_text':'\n'.join(lines),
    }
