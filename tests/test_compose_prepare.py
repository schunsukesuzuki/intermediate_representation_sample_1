"""Regression tests for v3 -> v3.1 Hetionet CSV preparation error.

Mocks upstream conversion/download, so tests run offline without Polars/Voyager.
"""
from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import pytest

from gnn_rag import compose_prepare


def _sample(directory: Path, *, focus="Alzheimer's disease", max_nodes=120, seed=42):
    directory.mkdir(parents=True, exist_ok=True)
    node_ids = ["Disease::DOID:10652"] + [f"Gene::{i}" for i in range(11)]
    with (directory / "nodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["node_id", "name", "kind"])
        w.writeheader()
        w.writerows({"node_id": x, "name": x, "kind": "Gene"} for x in node_ids)
    with (directory / "edges.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target", "relation"])
        w.writeheader()
        w.writerows({"source": node_ids[0], "target": x, "relation": "DaG"}
                    for x in node_ids[1:])
    meta = {"focus_id": node_ids[0], "focus_name": focus, "nodes": 12,
            "edges": 11, "requested_max_nodes": max_nodes, "seed": seed,
            "schema_version": 4, "triple_preserving": True}
    (directory / "dataset_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def test_synthetic_skips_all_data_download_and_conversion(tmp_path, monkeypatch):
    monkeypatch.setattr(compose_prepare, "ensure_hetionet_inputs", lambda *a, **kw: pytest.fail("download"))
    result = compose_prepare.prepare_data({"GNN_RAG_DATASET": "synthetic"}, data_root=tmp_path)
    assert result["status"] == "not_required"
    assert not list(tmp_path.iterdir())


def test_cached_prepared_csv_does_not_need_raw_files(tmp_path, monkeypatch):
    _sample(tmp_path / "hetionet_demo")
    monkeypatch.setattr(compose_prepare, "ensure_hetionet_inputs", lambda *a, **kw: pytest.fail("download"))
    result = compose_prepare.prepare_data({"GNN_RAG_DATASET": "hetionet"}, data_root=tmp_path)
    assert result["status"] == "cached"


def test_missing_csv_auto_downloads_then_converts(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    calls = []

    def fake_download(path, *, download):
        calls.append((path, download))
        return raw / "nodes.tsv", raw / "edges.sif.gz"

    def fake_convert(nodes, edges, output, *, focus, max_nodes, seed):
        calls.append((nodes, edges, focus, max_nodes, seed))
        return _sample(output, focus=focus, max_nodes=max_nodes, seed=seed)

    monkeypatch.setattr(compose_prepare, "ensure_hetionet_inputs", fake_download)
    monkeypatch.setitem(sys.modules, "gnn_rag.public_data", types.SimpleNamespace(convert_hetionet=fake_convert))
    result = compose_prepare.prepare_data({"GNN_RAG_DATASET": "hetionet"}, data_root=tmp_path)
    assert result["status"] == "created"
    assert calls[0] == (raw, True)
    assert calls[1][-3:] == ("Alzheimer's disease", 120, 42)
    assert (tmp_path / "hetionet_demo/nodes.csv").is_file()


def test_repeated_run_reuses_csv(tmp_path, monkeypatch):
    _sample(tmp_path / "hetionet_demo")
    def fail(*args, **kwargs):
        pytest.fail("Should not download or re-convert")
    monkeypatch.setattr(compose_prepare, "ensure_hetionet_inputs", fail)
    assert compose_prepare.prepare_data({"GNN_RAG_DATASET": "hetionet"}, data_root=tmp_path)["status"] == "cached"
    assert compose_prepare.prepare_data({"GNN_RAG_DATASET": "hetionet"}, data_root=tmp_path)["status"] == "cached"


def test_partial_or_corrupted_csv_is_not_trusted(tmp_path):
    output = tmp_path / "hetionet_demo"
    _sample(output)
    (output / "edges.csv").write_text("source,target,relation\nwrong,unknown,DrD\n", encoding="utf-8")
    okay, reason = compose_prepare.cached_hetionet_is_usable(output, focus="Alzheimer's disease", max_nodes=120, seed=42)
    assert not okay
    assert "counts" in reason or "edge" in reason


def test_parameter_change_requires_rebuild(tmp_path):
    output = tmp_path / "hetionet_demo"
    _sample(output)
    assert not compose_prepare.cached_hetionet_is_usable(output, focus="Parkinson's disease", max_nodes=120, seed=42)[0]
    assert not compose_prepare.cached_hetionet_is_usable(output, focus="Alzheimer's disease", max_nodes=125, seed=42)[0]
    assert not compose_prepare.cached_hetionet_is_usable(output, focus="Alzheimer's disease", max_nodes=120, seed=3)[0]


def test_legacy_v3_cache_is_rejected_for_triple_preservation(tmp_path):
    output = tmp_path / "hetionet_demo"
    metadata = _sample(output)
    for key in ("requested_max_nodes", "seed", "schema_version", "triple_preserving"):
        del metadata[key]
    (output / "dataset_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    ready, reason = compose_prepare.cached_hetionet_is_usable(output, focus="Alzheimer's disease", max_nodes=120, seed=42)
    assert not ready
    assert 'old pair-collapsed cache' in reason
    assert not compose_prepare.cached_hetionet_is_usable(output, focus="Alzheimer's disease", max_nodes=125, seed=42)[0]


def test_unsupported_dataset_fails_instead_of_synthetic_fallback(tmp_path):
    with pytest.raises(ValueError, match="synthetic or hetionet"):
        compose_prepare.prepare_data({"GNN_RAG_DATASET": "primekg"}, data_root=tmp_path)


def test_offline_mode_with_missing_raw_fails_informatively(tmp_path):
    with pytest.raises(FileNotFoundError, match="Hetionet file"):
        compose_prepare.prepare_data({"GNN_RAG_DATASET": "hetionet", "HETIONET_AUTO_DOWNLOAD": "0"}, data_root=tmp_path)


def test_accepted_settings_and_invalid_bounds(tmp_path):
    for env in [
        {"GNN_RAG_DATASET": "hetionet", "HETIONET_MAX_NODES": "601"},
        {"GNN_RAG_DATASET": "hetionet", "GNN_RAG_SEED": "-1"},
        {"GNN_RAG_DATASET": "hetionet", "HETIONET_AUTO_DOWNLOAD": "sometimes"},
    ]:
        with pytest.raises(ValueError):
            compose_prepare.prepare_data(env, data_root=tmp_path)


def test_corrupted_prepared_csv_is_regenerated(tmp_path, monkeypatch):
    output = tmp_path / "hetionet_demo"
    _sample(output)
    (output / "edges.csv").write_text("invalid,columns\n", encoding="utf-8")
    called = []

    def download(path, *, download):
        called.append("download_or_check")
        return tmp_path / "raw/nodes.tsv", tmp_path / "raw/edges.gz"

    def convert(nodes, edges, output, *, focus, max_nodes, seed):
        called.append("convert")
        return _sample(output, focus=focus, max_nodes=max_nodes, seed=seed)

    monkeypatch.setattr(compose_prepare, "ensure_hetionet_inputs", download)
    monkeypatch.setitem(sys.modules, "gnn_rag.public_data", types.SimpleNamespace(convert_hetionet=convert))
    outcome = compose_prepare.prepare_data({"GNN_RAG_DATASET": "hetionet"}, data_root=tmp_path)
    assert outcome["status"] == "created"
    assert called == ["download_or_check", "convert"]
    assert compose_prepare.cached_hetionet_is_usable(output, focus="Alzheimer's disease", max_nodes=120, seed=42)[0]


def test_raw_download_failure_does_not_fallback_to_synthetic(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr(compose_prepare, "ensure_hetionet_inputs", fail)
    with pytest.raises(RuntimeError, match="upstream unavailable"):
        compose_prepare.prepare_data({"GNN_RAG_DATASET": "hetionet"}, data_root=tmp_path)
    assert not (tmp_path / "hetionet_demo/nodes.csv").exists()
