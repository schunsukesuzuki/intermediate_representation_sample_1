"""v10: four independent second-stage explanations and PDF persistence.

No live Ollama required; these tests exercise the actual integration helpers,
prompt isolation, error handling and ReportLab output.
"""
from pathlib import Path

import pytest

from gnn_rag.four_way_explanation import add_explanations, FOUR_METHODS
from gnn_rag.llm import OllamaClient
from gnn_rag.comparison_pdf import generate_four_way_pdf, FOUR_WAY_PDF


def payload():
    entries = {}
    for method in FOUR_METHODS:
        entries[method] = {
            'answer': f'原回答 {method}：この関係がKGに登録されている。',
            'status': 'unverified_generation' if method == 'plain_qwen' else 'selected',
            'catalog': [] if method == 'plain_qwen' else [
                {'evidence_id': 'E1', 'statement': f'選択済みの根拠 {method}'},
                {'evidence_id': 'E2', 'statement': '未選択の根拠'},
            ],
            'selected_evidence_ids': [] if method == 'plain_qwen' else ['E1'],
            'evidence': [],
        }
    return {
        'created_utc': '2026-11-15T04:00:00+00:00',
        'node_name': 'アルツハイマー病', 'node_id': 'Disease::DOID:10652',
        'question': '登録された関係を説明してください', 'model': 'qwen3:1.7b',
        'hops': 2, 'top_k': 3, 'fusion_alpha': 0.5,
        'fusion_calibrated': False, 'results': entries,
    }


def test_exactly_one_separate_qwen_explanation_per_original_answer():
    seen = []
    class MockClient:
        def explain_result(self, **kwargs):
            seen.append(kwargs)
            return {'sequence_interpretation': '関係仮説：' + kwargs['method'],
                    'explanation': '説明：' + kwargs['method']}
    report = payload()
    old_answers = {k: report['results'][k]['answer'] for k in FOUR_METHODS}
    add_explanations(report, MockClient())
    assert [x['method'] for x in seen] == list(FOUR_METHODS)
    for method in FOUR_METHODS:
        entry = report['results'][method]
        assert entry['answer'] == old_answers[method]
        assert entry['explanation'] == '説明：' + method
        assert entry['sequence_interpretation'] == '関係仮説：' + method
        assert entry['explanation_status'] == 'generated_unverified'
    assert not seen[0]['catalog']
    assert not seen[0]['selected_evidence_ids']
    assert all(len(x['catalog']) == 2 for x in seen[1:])
    assert 'unverified' in report['explanation_note']


def test_independent_error_does_not_erase_other_outputs():
    class MockClient:
        def explain_result(self, **kwargs):
            if kwargs['method'] == 'gnn_only':
                raise RuntimeError('Ollama temporarily unavailable')
            return {'sequence_interpretation': '仮説・未検証', 'explanation': '説明が生成された'}
    report = payload()
    report['results']['bfs_only']['answer'] = ''
    add_explanations(report, MockClient())
    assert report['results']['bfs_only']['explanation_status'] == 'skipped_no_answer'
    assert report['results']['gnn_only']['explanation_status'] == 'error'
    assert report['results']['gnn_only']['explanation_error'] == 'Ollama temporarily unavailable'
    assert report['results']['bfs_gnn']['explanation_status'] == 'generated_unverified'
    assert report['results']['gnn_only']['answer'].startswith('原回答')


def test_second_stage_prompt_contains_no_kg_for_plain_qwen(monkeypatch):
    captured = []
    def fake_chat(self, prompt, **kwargs):
        captured.append((prompt, kwargs))
        return '{"sequence_interpretation":"KGシーケンス未提供","explanation":"この回答は未検証です。"}'
    monkeypatch.setattr(OllamaClient, '_chat', fake_chat)
    client = OllamaClient()
    catalog = [{'evidence_id':'E1','statement':'選択済みの根拠A'},
               {'evidence_id':'E2','statement':'未選択の根拠B'}]
    for method in FOUR_METHODS:
        text = client.explain_result(method=method, subject='Node', question='Q?',
            answer='原回答', answer_status='selected', catalog=catalog,
            selected_evidence_ids=['E1'])
        assert text['explanation'] == 'この回答は未検証です。'
    assert len(captured) == 4
    plain_prompt, plain_kwargs = captured[0]
    assert '選択済みの根拠A' not in plain_prompt
    assert '未選択の根拠B' not in plain_prompt
    assert 'NOT PROVIDED' in plain_prompt
    for prompt, options in captured[1:]:
        assert '選択済みの根拠A' in prompt
        assert '未選択の根拠B' not in prompt
        assert options['max_output_tokens'] >= 400
    assert 'KG' in plain_kwargs['system_prompt']


def test_four_way_pdf_includes_explanations(tmp_path: Path):
    class MockClient:
        def explain_result(self, **kwargs):
            return {'sequence_interpretation': '仮説・未検証',
                    'explanation': 'Qwenによる説明文：' + kwargs['method']}
    data = add_explanations(payload(), MockClient())
    path = generate_four_way_pdf(tmp_path / FOUR_WAY_PDF, data)
    assert path.read_bytes()[:5] == b'%PDF-'
    assert path.stat().st_size > 4000
    assert not path.with_suffix('.pdf.tmp').exists()
    assert all(data['results'][k]['explanation'].startswith('Qwenによる') for k in FOUR_METHODS)
