"""v14 regression: schema, retry, local raw diagnostics, provenance and truncation."""
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request
import pytest
from gnn_rag.llm import OllamaClient
from gnn_rag.llm_json import StructuredOutputError
from gnn_rag.four_way_explanation import add_explanations, add_cross_comparison
from gnn_rag.structured_paths import build_cross_factual_comparison, build_method_audit
from gnn_rag.comparison_pdf import generate_four_way_pdf


def sample():
    rec={}
    for k in ('plain_qwen','bfs_only','gnn_only','bfs_gnn'):
        rec[k]={'answer':'登録情報を調べる '+k,'status':'unverified_generation' if k=='plain_qwen' else 'selected',
                'selected_evidence_ids':[] if k=='plain_qwen' else ['E1'],
                'catalog':[] if k=='plain_qwen' else [
                    dict(evidence_id='E1',source='D',target='A',relation='DlA',source_name='疾患',target_name='脳',
                         source_kind='Disease',target_kind='Anatomy',label='抄録共起',provenance='共起',statement='疾患と脳の共起'),
                    dict(evidence_id='E2',source='A',target='G',relation='AeG',source_name='脳',target_name='遺伝子',
                         source_kind='Anatomy',target_kind='Gene',label='遺伝子発現',provenance='発現',statement='脳での遺伝子発現')],
                'evidence':[] if k=='plain_qwen' else [dict(nodes=['D','A','G'],edges=[
                    dict(source='D',relation='DlA',target='A'),dict(source='A',relation='AeG',target='G')])],
               }
    return dict(results=rec,node_name='疾患',node_id='D',question='登録した関係は?',hops=2,top_k=3,
                model='qwen3:1.7b',fusion_alpha=0.5,fusion_calibrated=True,dataset='hetionet_subset')


def example_explanation():
    return json.dumps({'sequence_interpretation':'P1のDlAとAeGは異なる型の登録関係。未検証の仮説。',
                       'explanation':'この回答は登録エッジを参照。因果は証明しません。'},ensure_ascii=False)


def test_invalid_json_first_response_is_retried_shorter(monkeypatch):
    requests=[]
    def fake(self,prompt,**opts):
        requests.append((prompt,opts))
        return '{"sequence_interpretation":"P1","explanation":"' if len(requests)==1 else example_explanation()
    monkeypatch.setattr(OllamaClient,'_chat',fake)
    item=OllamaClient().explain_result(method='bfs_only',subject='疾患',question='Q?',answer='原回答',
            answer_status='selected',catalog=sample()['results']['bfs_only']['catalog'],
            selected_evidence_ids=['E1'],evidence_paths=sample()['results']['bfs_only']['evidence'])
    assert len(requests)==2
    assert len(requests[1][0])<=len(requests[0][0])
    assert item['generation_warning'].startswith('Recovered')
    assert requests[0][1]['schema']['required']==['sequence_interpretation','explanation']


def test_plain_is_truly_kg_isolated_in_prompts(monkeypatch):
    prompts=[]
    def fake(self,prompt,**opts):
        prompts.append(prompt)
        return example_explanation()
    monkeypatch.setattr(OllamaClient,'_chat',fake)
    add_explanations(sample(),OllamaClient())
    assert 'NOT PROVIDED (plain Qwen baseline)' in prompts[0]
    assert '疾患と脳の共起' not in prompts[0]
    assert 'plain_qwenではKGのシーケンス' not in prompts[1]
    assert '未選択だからエッジが存在しない' in prompts[1]
    assert 'relation_code' in prompts[1]


def test_repeated_bad_json_preserves_raw_and_error(monkeypatch,tmp_path):
    monkeypatch.setattr(OllamaClient,'_chat',lambda self,prompt,**opts:'{"sequence_interpretation":"not done"')
    data=sample()
    add_explanations(data,OllamaClient())
    item=data['results']['bfs_only']
    assert item['explanation_status']=='error'
    assert len(item['explanation_attempts'])==2
    assert 'not done' in item['explanation_raw_response']
    assert 'not done' not in item['explanation_error']
    assert 'P1' in item['structured_path_text']
    pdf=generate_four_way_pdf(tmp_path/'v14.pdf',data)
    from pypdf import PdfReader
    doc='\n'.join(p.extract_text() or '' for p in PdfReader(str(pdf)).pages)
    assert 'GNN-RAG v14' in doc
    assert '原文' not in doc or 'JSON' in doc


def test_misdeclared_no_path_is_retried_and_rejected(monkeypatch):
    monkeypatch.setattr(OllamaClient,'_chat',lambda *args,**kwargs:
        json.dumps({'sequence_interpretation':'KGシーケンス未提供', 'explanation':'KGなし'},ensure_ascii=False))
    with pytest.raises(StructuredOutputError,match='incorrectly denied'):
        OllamaClient().explain_result(method='gnn_only',subject='疾患',question='?',answer='回答',
            answer_status='selected',catalog=sample()['results']['gnn_only']['catalog'],
            selected_evidence_ids=['E1'],evidence_paths=sample()['results']['gnn_only']['evidence'])


def test_cross_facts_are_exact_triples_not_local_evidence_ids():
    d=sample()['results']
    d['gnn_only']['catalog'][0]['relation']='AuG'
    d['bfs_gnn']['selected_evidence_ids']=['E2']
    audit=build_method_audit(d)
    summary=build_cross_factual_comparison(audit)
    assert summary['plain_qwen_kg_edges']==0
    assert len(summary['common_triples'])==0
    assert summary['selected_counts']=={'bfs_only':1,'gnn_only':1,'bfs_gnn':1}
    assert len(summary['unique_triples']['bfs_only'])==1
    assert 'AeG' in summary['summary_text'] and 'AuG' in summary['summary_text']


def test_cross_prompt_is_schema_enforced_and_only_hypotheses(monkeypatch):
    seen=[]
    def fake(self,prompt,**opts):
        seen.append((prompt,opts))
        return json.dumps({'interpretive_observation':'説明内容の強調点に差があるかもしれない。',
                           'limitations':'各方法の医学的正しさは未確認。'},ensure_ascii=False)
    monkeypatch.setattr(OllamaClient,'_chat',fake)
    data=sample()
    for k,v in data['results'].items():
        v.update(sequence_interpretation='仮説', explanation='事実ではない',explanation_status='generated_unverified')
    add_cross_comparison(data,OllamaClient())
    assert data['cross_comparison_status']=='generated_unverified'
    assert data['cross_factual_comparison']['plain_qwen_kg_edges']==0
    assert '根拠の限界' in data['cross_comparison']
    assert seen[0][1]['schema']['required']==['interpretive_observation','limitations']
    assert 'has_registered_KG_evidence' in seen[0][0]
    assert 'selected_train_edges' not in seen[0][0]


def test_ollama_truncation_detection(monkeypatch):
    class Resp:
        def __enter__(self):return self
        def __exit__(self,*x):return False
        def read(self):return b'{"message":{"content":"{\\"explanation\\":\\"cut\\""},"done_reason":"length"}'
    monkeypatch.setattr('gnn_rag.llm.urlopen',lambda *args,**kwargs:Resp())
    with pytest.raises(StructuredOutputError,match='generation limit'):
        OllamaClient()._chat('hello',schema={'type':'object'},max_output_tokens=60)


def test_frontend_and_json_endpoint_persisted():
    root=Path(__file__).resolve().parents[1]
    assert 'RESEARCH CONSOLE <b>v14</b>' in (root/'frontend/src/App.tsx').read_text()
    assert 'cross_factual_comparison?.summary_text' in (root/'frontend/src/App.tsx').read_text()
    assert "@api.get('/comparison-json')" in (root/'src/gnn_rag/api.py').read_text()
