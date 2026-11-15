"""v13 regression: deterministic typed paths, provenance and isolation."""
from __future__ import annotations

import json
from pathlib import Path

from gnn_rag.structured_paths import build_verified_paths, format_verified_paths, build_method_audit
from gnn_rag.four_way_explanation import add_explanations, add_cross_comparison
from gnn_rag.llm import OllamaClient
from gnn_rag.comparison_pdf import generate_four_way_pdf


def catalog():
    return [
        dict(evidence_id='E1', source='D', target='A', source_name="Alzheimer's disease",
             target_name='brain', source_kind='Disease', target_kind='Anatomy',
             relation='DlA', label='疾患と解剖学的部位の文献共起', provenance='抄録共起',
             statement='疾患と解剖部位の登録関係'),
        dict(evidence_id='E2', source='A', target='G', source_name='brain',
             target_name='APOE', source_kind='Anatomy', target_kind='Gene',
             relation='AeG', label='解剖学的部位での遺伝子発現', provenance='遺伝子発現データ',
             statement='解剖部位での発現の登録関係'),
    ]


def backward_path():
    return [dict(nodes=['G', 'A', 'D'], edges=[
        dict(source='A', target='G', relation='AeG'),
        dict(source='D', target='A', relation='DlA'),
    ])]


def test_reverse_traversal_preserves_canonical_typed_edges_and_unselected():
    paths = build_verified_paths(catalog(), backward_path(), ['E2'])
    assert len(paths) == 1 and len(paths[0]['steps']) == 2
    a, b = paths[0]['steps']
    assert a['relation_code'] == 'AeG' and a['traversal_direction'] == 'reverse'
    assert a['source_name'] == 'brain' and a['target_name'] == 'APOE'
    assert b['relation_code'] == 'DlA' and b['traversal_direction'] == 'reverse'
    assert b['selected_for_answer'] is False and b['observed_train_edge'] is True
    assert '抄録共起' in format_verified_paths(paths)
    assert '遺伝子発現データ' in format_verified_paths(paths)
    assert '未採用（エッジは存在）' in format_verified_paths(paths)


def test_unknown_middle_edge_never_reattaches_suffix():
    path = dict(nodes=['D', 'A', 'X', 'G'], edges=[
        dict(source='D', relation='DlA', target='A'),
        dict(source='A', relation='NOT_TRAIN', target='X'),
        dict(source='A', relation='AeG', target='G'),
    ])
    out = build_verified_paths(catalog(), [path], ['E1'])
    assert len(out[0]['steps']) == 1
    assert 'X' not in json.dumps(out)
    assert 'NOT_TRAIN' not in json.dumps(out)


def test_mismatched_nodes_never_produce_a_fabricated_path():
    bad = [dict(nodes=['D', 'G'], edges=[dict(source='D', relation='DlA', target='A')])]
    assert build_verified_paths(catalog(), bad, ['E1']) == []


def sample_report():
    records = {}
    for key in ('plain_qwen', 'bfs_only', 'gnn_only', 'bfs_gnn'):
        records[key] = dict(answer='原回答 '+key, status='selected',
                            catalog=catalog() if key != 'plain_qwen' else [],
                            evidence=backward_path() if key != 'plain_qwen' else [],
                            selected_evidence_ids=['E2'] if key != 'plain_qwen' else [])
    return dict(node_name="Alzheimer's disease",node_id='D', question='関係は?',
                model='qwen3:1.7b', dataset='hetionet_subset',hops=2,top_k=3,
                fusion_alpha=.5, fusion_calibrated=True,results=records)


def test_plain_qwen_never_receives_kg_in_explanation_or_audit(monkeypatch):
    prompts = []
    def fake_chat(self, prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(dict(sequence_interpretation='P1は未検証仮説として解釈',
                               explanation='原回答は未検証です。'), ensure_ascii=False)
    monkeypatch.setattr(OllamaClient, '_chat', fake_chat)
    report = sample_report()
    add_explanations(report, OllamaClient())
    assert 'DlA' not in prompts[0] and 'AeG' not in prompts[0]
    assert report['results']['plain_qwen']['structured_paths'] == []
    for prompt in prompts[1:]:
        assert '"relation_code":"DlA"' in prompt
        assert '"relation_code":"AeG"' in prompt
        assert '遺伝子発現データ' in prompt and '抄録共起' in prompt
    audit = build_method_audit(report['results'])
    assert audit['plain_qwen']['selected_train_edges'] == []
    assert audit['bfs_only']['selected_train_edges'][0]['relation'] == 'AeG'
    assert len(audit['bfs_only']['selected_train_edges']) == 1


def test_unsupported_llm_no_path_statement_replaced_and_pdf_preserves_paths(monkeypatch, tmp_path:Path):
    def fake_chat(self, prompt, **kwargs):
        if 'four_methods' in prompt:
            return '各方式を区別した定性的な比較（未検証）。'
        return json.dumps(dict(sequence_interpretation='KGシーケンス未提供',
                               explanation='根拠には限界があります。'), ensure_ascii=False)
    monkeypatch.setattr(OllamaClient, '_chat', fake_chat)
    report = sample_report()
    add_explanations(report, OllamaClient())
    assert report['results']['plain_qwen']['sequence_interpretation'] == 'KGシーケンス未提供'
    for key in ('bfs_only','gnn_only','bfs_gnn'):
        result=report['results'][key]
        assert result['explanation_status'] == 'error' and 'incorrectly denied' in result['explanation_error']
        assert 'AeG' in result['structured_path_text'] and 'DlA' in result['structured_path_text']
    add_cross_comparison(report, OllamaClient())
    assert report['method_audit']['plain_qwen']['source'] == 'no_kg_retrieval'
    pdf = generate_four_way_pdf(tmp_path/'comparison.pdf', report)
    assert pdf.stat().st_size > 4000
    from pypdf import PdfReader
    text='\n'.join(p.extract_text() or '' for p in PdfReader(str(pdf)).pages)
    assert 'GNN-RAG v14' in text
    assert 'Deterministic provenance audit' in text
    assert 'AeG' in text and 'DlA' in text
