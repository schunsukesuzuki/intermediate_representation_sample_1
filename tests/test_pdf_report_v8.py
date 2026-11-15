"""Offline v8 PDF regression: report uses current-run metrics, no old figures."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gnn_rag.dual_scale_benchmark import benchmark, parse_args, summarize_runs
from gnn_rag.pdf_report import REPORT_NAME, generate_pdf_report


def _report(seed: int, scale: int) -> dict:
    def metric(v):
        return {'test': {k: float(v if k in ('hits@10', 'mrr@10') else .12)
                         for k in ('hits@1','hits@3','hits@5','hits@10','recall@10','mrr@10')}}
    return {'seed': seed, 'node_count': scale//20,
            'candidate_counts_test': {'median': scale//100},
            'methods': {'gnn_only': metric(.2), 'bfs_only': metric(.1), 'bfs_gnn': metric(.27)},
            'calibration': {'alpha': .65},
            'training_metrics': {'train_loss': .78, 'train_triples': int(scale*.7)},
            'runtime_seconds': 10.0}


def test_pdf_current_run_two_scales(tmp_path: Path):
    results = {str(s): summarize_runs([_report(42,s),_report(43,s)],size=s)
               for s in (10000,100000)}
    payload = {'parameters': {'sample_seed':42,'seeds':[42,43], 'epochs':2,
               'max_positives_per_epoch':512,'eval_max_queries':16,
               'feature_mode':'kind_name','created_utc':'2026-11-15T01:00:00+00:00'},
               'sizes':results}
    path = generate_pdf_report(tmp_path/REPORT_NAME,payload)
    assert path.read_bytes().startswith(b'%PDF-')
    assert path.stat().st_size > 5000
    assert not (tmp_path/(REPORT_NAME+'.tmp')).exists()


def test_cli_pdf_path_override():
    parsed = parse_args(['--pdf-output','/app/project_root/' + REPORT_NAME])
    assert parsed.pdf_output == Path('/app/project_root') / REPORT_NAME


def test_benchmark_writes_pdf_to_root_after_training(tmp_path: Path, monkeypatch):
    from gnn_rag import dual_scale_benchmark as job
    monkeypatch.setattr(job, 'prepare_samples',lambda *a,**kw: (
        {10000:tmp_path,100000:tmp_path}, {'source_mode':'test'}))
    monkeypatch.setattr(job, 'run_one', lambda dataset, destination, *, seed, **kw:
                        _report(seed, 10000 if '10000' in str(destination) else 100000))
    params = parse_args(['--data-root',str(tmp_path/'data'),
                         '--output-root',str(tmp_path/'outputs'/'benchmark'),
                         '--pdf-output',str(tmp_path/'root'/REPORT_NAME),
                         '--sizes','10000','100000','--seeds','42','--epochs','1'])
    benchmark(params)
    assert (tmp_path/'root'/REPORT_NAME).read_bytes().startswith(b'%PDF-')
    assert (tmp_path/'outputs'/'benchmark'/REPORT_NAME).read_bytes() == (tmp_path/'root'/REPORT_NAME).read_bytes()


def test_incomplete_study_does_not_leave_previous_pdf(tmp_path: Path, monkeypatch):
    from gnn_rag import dual_scale_benchmark as job
    root = tmp_path / 'root' / REPORT_NAME
    root.parent.mkdir()
    root.write_bytes(b'old benchmark, must not be mislabeled as current')
    monkeypatch.setattr(job, 'prepare_samples', lambda *a, **kw: (
        {10000: tmp_path, 100000: tmp_path}, {'source_mode':'test'}))
    def broken(dataset, destination, *, seed, **kwargs):
        if '100000' in str(destination):
            raise RuntimeError('training failed')
        return _report(seed, 10000)
    monkeypatch.setattr(job, 'run_one', broken)
    args = parse_args(['--data-root',str(tmp_path/'data'),
                       '--output-root',str(tmp_path/'outputs'),
                       '--pdf-output',str(root), '--sizes','10000','100000','--seeds','42'])
    with pytest.raises(RuntimeError, match='training failed'):
        benchmark(args)
    assert not root.exists()
    assert not (tmp_path/'outputs'/REPORT_NAME).exists()
