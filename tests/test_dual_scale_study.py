"""Offline tests: ZIP safety, no implicit downloads, caching and reports."""
import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from gnn_rag.dataset_archive import resolve_hetionet_source
from gnn_rag.dual_scale_benchmark import (parse_args, summarize_runs,
                                          write_overview, prepare_samples)


def test_cli_defaults():
    opts = parse_args([])
    assert opts.sizes == [10000, 100000]
    assert opts.seeds == [42,43,44,45,46]
    assert opts.feature_mode == 'kind_name'


def test_absent_data_no_network(tmp_path):
    with pytest.raises(FileNotFoundError, match='Place hetionet'):
        resolve_hetionet_source(tmp_path, download=False)


def test_invalid_archive_does_not_silently_download(tmp_path):
    d = tmp_path / 'archives'; d.mkdir()
    with ZipFile(d / 'hetionet-v1.0.0.zip', 'w') as z:
        z.writestr('foo.txt', 'bad file')
    with pytest.raises(RuntimeError, match='Invalid Hetionet archive'):
        resolve_hetionet_source(tmp_path, download=True)


def test_summary_and_markdown(tmp_path):
    def score(v):
        return {'test': {key: (v if key in ('hits@10', 'mrr@10') else 0.1)
                         for key in ('hits@1', 'hits@3', 'hits@5', 'hits@10', 'recall@10', 'mrr@10')}}
    reports = [
        {'seed': 42, 'node_count': 100, 'candidate_counts_test': {'median': 23},
         'methods': {'gnn_only': score(.2), 'bfs_only': score(.1), 'bfs_gnn': score(.25)},
         'calibration': {'alpha': .8}},
        {'seed': 43, 'node_count': 102, 'candidate_counts_test': {'median': 25},
         'methods': {'gnn_only': score(.4), 'bfs_only': score(.2), 'bfs_gnn': score(.45)},
         'calibration': {'alpha': .9}},
    ]
    summary = summarize_runs(reports, size=10000)
    assert summary['methods']['gnn_only']['mrr@10']['mean'] == pytest.approx(.3)
    assert summary['methods']['gnn_only']['mrr@10']['sd'] > 0
    assert summary['alphas'] == [.8, .9]
    write_overview(tmp_path, {'10000': summary},
                   {'sample_seed': 42, 'seeds': [42, 43], 'epochs': 25,
                    'max_positives_per_epoch': 4096, 'eval_max_queries': 128})
    assert '| 10,000 | 101 |' in (tmp_path / 'overview.md').read_text()
    assert len(list(csv.DictReader((tmp_path / 'summary.csv').open()))) == 3
    assert json.loads((tmp_path / 'overview.json').read_text())['sizes']['10000']['num_seeds'] == 2
