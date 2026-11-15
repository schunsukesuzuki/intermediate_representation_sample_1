"""v12 migration smoke: Metal host, async comparison, dataset provenance.

Does not require a running Ollama or internet access.
"""
from pathlib import Path
from threading import Event
from time import monotonic, sleep
import json

import pytest
from gnn_rag.comparison_jobs import ComparisonJobs
from gnn_rag.llm import OllamaClient
from gnn_rag.four_way_explanation import add_explanations


def _await_status(jobs, job_id, desired, seconds=3):
    until = monotonic() + seconds
    while monotonic() < until:
        state = jobs.status(job_id)
        if state['status'] == desired:
            return state
        sleep(.01)
    raise AssertionError(f'Job never became {desired}: {state}')


def test_comparison_async_lifecycle_progress_and_result():
    gate = Event()
    jobs = ComparisonJobs()
    def task(progress):
        progress('plain_qwen')
        gate.wait(2)
        progress('saving_report')
        return {'pdf_name': 'GNN_RAG_four_way_answers.pdf'}
    job_id = jobs.submit(task)
    _await_status(jobs, job_id, 'running')
    assert jobs.status(job_id)['stage'] == 'plain_qwen'
    with pytest.raises(RuntimeError, match='already running'):
        jobs.submit(task)
    gate.set()
    state = _await_status(jobs, job_id, 'completed')
    assert state['result']['pdf_name'].endswith('.pdf')
    assert state['stage'] == 'done'
    assert jobs.status('not-a-job') is None


def test_comparison_async_errors_preserve_failure_state():
    jobs = ComparisonJobs()
    def task(progress):
        progress('cross_comparison')
        raise RuntimeError('model unavailable')
    job_id = jobs.submit(task)
    failed = _await_status(jobs, job_id, 'failed')
    assert failed['stage'] == 'failed'
    assert 'model unavailable' in failed['error']


def test_native_host_llm_endpoint_accepted_and_context_is_explicit(monkeypatch):
    called = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, *_): return b'{"message":{"content":"ok"}}'
    def fetch(request, timeout):
        called['url'] = request.full_url
        called['payload'] = json.loads(request.data)
        called['timeout'] = timeout
        return Response()
    monkeypatch.setattr('gnn_rag.llm.urlopen', fetch)
    monkeypatch.setenv('GNN_RAG_OLLAMA_NUM_CTX', '8192')
    assert OllamaClient(endpoint='http://host.docker.internal:11434', timeout=900).answer('test') == 'ok'
    assert called['url'] == 'http://host.docker.internal:11434/api/chat'
    assert called['payload']['options']['num_ctx'] == 8192
    assert called['timeout'] == 900
    for url in ('http://example.com:11434','http://host.docker.internal:12345','https://host.docker.internal:11434'):
        with pytest.raises(ValueError):
            OllamaClient(endpoint=url).answer('test')


def test_unselected_train_edges_are_not_absent(monkeypatch):
    prompts=[]
    def fake_chat(self, prompt, **kwargs):
        prompts.append(prompt)
        return '{"sequence_interpretation":"未検証の仮説","explanation":"非選択のエッジも登録済みです。"}'
    monkeypatch.setattr(OllamaClient,'_chat', fake_chat)
    report = OllamaClient().explain_result(
        method='gnn_only',subject='看護記録',question='登録された関係は?',
        answer='観測済みの関係',answer_status='selected',
        catalog=[{'source':'n0','target':'n1','relation':'related_to','evidence_id':'E1','statement':'登録済みn0-n1'},
                 {'source':'n1','target':'n2','relation':'related_to','evidence_id':'E2','statement':'登録済みn1-n2'}],
        selected_evidence_ids=['E1'],
        evidence_paths=[{'nodes':['n0','n1','n2'],'edges':[
            {'source':'n0','relation':'related_to','target':'n1'},
            {'source':'n1','relation':'related_to','target':'n2'}]}],
    )
    assert report['explanation']
    assert 'retrieved_sequences_present' in prompts[0]
    assert '未選択だからエッジが存在しない' in prompts[0]
    assert '登録済みn1-n2' in prompts[0]  # retrieved but not selected


def test_metal_compose_and_cpu_fallback_are_separate():
    root=Path(__file__).resolve().parents[1]
    main=(root/'compose.yaml').read_text()
    fallback=(root/'compose.cpu.yaml').read_text()
    assert '  ollama:\n' not in main
    assert '  model-pull:\n' not in main
    assert 'http://host.docker.internal:11434' in main
    assert '  ollama:\n' in fallback
    assert '  model-pull:\n' in fallback
    assert 'GNN_RAG_OLLAMA_URL: http://ollama:11434' in fallback
    assert 'GNN_RAG_DATASET=hetionet' in (root/'.env.example').read_text()


def test_metal_pdf_metadata_and_frontend_status():
    root=Path(__file__).resolve().parents[1]
    assert 'Dataset:' in (root/'src/gnn_rag/comparison_pdf.py').read_text()
    assert '/compare-jobs' in (root/'frontend/src/App.tsx').read_text()
    assert 'Synthetic demo graph' in (root/'frontend/src/App.tsx').read_text()
    assert 'version="0.14.0"' in (root/'src/gnn_rag/api.py').read_text()
