"""v11: exact input sequences, safe boundaries, Qwen cross-method synthesis, PDF."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

from gnn_rag.llm import OllamaClient
from gnn_rag.four_way_explanation import FOUR_METHODS, add_cross_comparison, add_explanations
from gnn_rag.comparison_pdf import generate_four_way_pdf, FOUR_WAY_PDF


def payload():
    reports = {}
    for method in FOUR_METHODS:
        reports[method] = {
            'answer': 'original ' + method, 'status': 'selected',
            'catalog': [] if method == 'plain_qwen' else [
                {'evidence_id': 'E1', 'source': 'A', 'relation': 'Rel', 'target': 'B',
                 'source_name': '甲', 'target_name': '乙', 'statement': '甲と乙の登録リンク'},
                {'evidence_id': 'E2', 'source': 'B', 'relation': 'Rel', 'target': 'C',
                 'source_name': '乙', 'target_name': '丙', 'statement': '乙と丙の登録リンク'},
            ],
            'selected_evidence_ids': [] if method == 'plain_qwen' else ['E1'],
            'evidence': [] if method == 'plain_qwen' else [
                {'nodes': ['A', 'B', 'C'],
                 'edges': [
                     {'source': 'A', 'relation': 'Rel', 'target': 'B'},
                     {'source': 'B', 'relation': 'Rel', 'target': 'C'},
                     {'source': 'X', 'relation': 'UNSEEN_TEST', 'target': 'Y'},
                 ]},
            ],
        }
    return {'node_name': '甲', 'node_id': 'A', 'question': 'どう関係する?',
            'model': 'qwen3:1.7b', 'hops': 2, 'top_k': 3,
            'fusion_alpha': 0.5, 'fusion_calibrated': False,
            'results': reports}


def test_prompt_uses_actual_paths_and_plain_is_isolated(monkeypatch):
    captured = []
    def fake_chat(self, prompt, **opts):
        captured.append((prompt, opts))
        return json.dumps({'sequence_interpretation': '仮説・未検証',
                           'explanation': '登録情報からの解釈です。' * 3}, ensure_ascii=False)
    monkeypatch.setattr(OllamaClient, '_chat', fake_chat)
    report = add_explanations(payload(), OllamaClient())
    assert len(captured) == 4
    assert 'まず、得られたシーケンスをもとに、ノードとエッジから' in captured[0][0]
    plain = captured[0][0]
    assert '甲と乙の登録リンク' not in plain
    assert 'UNSEEN_TEST' not in plain
    assert 'NOT PROVIDED (plain Qwen baseline)' in plain
    for prompt, options in captured[1:]:
        assert '甲と乙の登録リンク' in prompt
        assert '乙と丙の登録リンク' in prompt
        assert 'selected_for_answer' in prompt
        assert 'UNSEEN_TEST' not in prompt  # an edge not type-validated in catalog
        assert options['schema']['required'] == ['sequence_interpretation', 'explanation']
    for method in FOUR_METHODS:
        assert report['results'][method]['sequence_interpretation'] == '仮説・未検証'
        assert report['results'][method]['explanation_status'] == 'generated_unverified'


def test_cross_comparison_happens_after_four_and_is_persisted(monkeypatch, tmp_path: Path):
    calls = []
    def fake_chat(self, prompt, **opts):
        if 'four_methods' in prompt:
            calls.append('cross')
            for key in FOUR_METHODS:
                assert 'original '+key in prompt
            assert '仮説の推測' in prompt or '未検証の仮説' in prompt
            assert 'has_registered_KG_evidence' in prompt
            return json.dumps({'interpretive_observation':'共通点: 甲を扱う。相違: BFS・GNNには登録リンクがある。', 'limitations':'正確性は未検証。'},ensure_ascii=False)
        method = FOUR_METHODS[len(calls)]
        calls.append(method)
        return json.dumps({'sequence_interpretation':'未検証の仮説',
                           'explanation':'登録されたリンクは因果の証明ではありません。'},ensure_ascii=False)
    monkeypatch.setattr(OllamaClient, '_chat', fake_chat)
    result = payload()
    add_explanations(result, OllamaClient())
    add_cross_comparison(result, OllamaClient())
    assert calls == list(FOUR_METHODS) + ['cross']
    assert result['cross_comparison_status'] == 'generated_unverified'
    assert '共通点' in result['cross_comparison']
    pdf = generate_four_way_pdf(tmp_path / FOUR_WAY_PDF, result)
    assert pdf.read_bytes().startswith(b'%PDF-')
    assert pdf.stat().st_size > 4500


def test_comparison_exception_does_not_erase_individual_outputs(tmp_path: Path):
    class Fake:
        def explain_result(self, **kwargs):
            return {'sequence_interpretation':'仮説・未検証', 'explanation':'個別説明'}
        def compare_four_explanations(self, **kwargs):
            assert len(kwargs['results']) == 4
            raise RuntimeError('Ollama timeout')
    data = add_explanations(payload(), Fake())
    add_cross_comparison(data, Fake())
    assert data['cross_comparison_status'] == 'error'
    assert data['cross_comparison_error'] == 'Ollama timeout'
    assert all(data['results'][k]['explanation'] == '個別説明' for k in FOUR_METHODS)
    output = generate_four_way_pdf(tmp_path / FOUR_WAY_PDF, data)
    assert output.is_file()


def test_bad_structured_explanation_rejected(monkeypatch):
    monkeypatch.setattr(OllamaClient, '_chat', lambda self, prompt, **kwargs:
                        json.dumps({'explanation':'missing sequence'}))
    with pytest.raises(RuntimeError, match='required'):
        OllamaClient().explain_result(method='plain_qwen', subject='x', question='y',
                                     answer='z', answer_status='selected',catalog=[],
                                     selected_evidence_ids=[])


def test_comparison_only_selected_facts_reach_cross_llm(monkeypatch):
    captured = []
    def fake_chat(self,prompt,**kw):
        captured.append(prompt)
        return json.dumps({'interpretive_observation':'横断比較（未検証）', 'limitations':'医学的正確性は未検証。'},ensure_ascii=False)
    monkeypatch.setattr(OllamaClient, '_chat', fake_chat)
    data = payload()
    for key in FOUR_METHODS:
        data['results'][key]['sequence_interpretation'] = '未検証の仮説'
        data['results'][key]['explanation'] = '個別説明'
        data['results'][key]['explanation_status'] = 'generated_unverified'
    add_cross_comparison(data, OllamaClient())
    assert len(captured) == 1
    assert 'four_methods' in captured[0]
    assert '乙と丙の登録リンク' not in captured[0]
    assert 'UNSEEN_TEST' not in captured[0]


def test_no_answers_skips_comparison():
    class Fake:
        def compare_four_explanations(self, **kwargs):
            raise AssertionError('Must not call LLM')
    data = payload()
    for key in FOUR_METHODS: data['results'][key]['answer'] = ''
    add_cross_comparison(data, Fake())
    assert data['cross_comparison_status'] == 'skipped_no_answers'
