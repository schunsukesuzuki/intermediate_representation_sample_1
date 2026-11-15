"""Config-boundary tests; no Docker engine or model download required."""
from pathlib import Path

import pytest

from gnn_rag import compose_train


def test_synthetic_defaults():
    options = compose_train.training_options({})
    assert options["epochs"] == 25
    assert options["seed"] == 42
    assert options["start_id"] == "n00"
    assert "nodes_csv" not in options


def test_invalid_dataset_does_not_silently_train_synthetic():
    with pytest.raises(ValueError, match="synthetic.*hetionet"):
        compose_train.training_options({"GNN_RAG_DATASET": "wikidata"})


def test_hetionet_requires_prepared_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(compose_train, "DEFAULT_DATA_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare-data"):
        compose_train.training_options({"GNN_RAG_DATASET": "hetionet"})
    (tmp_path / "nodes.csv").write_text("node_id\n", encoding="utf-8")
    (tmp_path / "edges.csv").write_text("source,target\n", encoding="utf-8")
    cfg = compose_train.training_options(
        {"GNN_RAG_DATASET": "hetionet", "GNN_RAG_EPOCHS": "7", "GNN_RAG_SEED": "1"}
    )
    assert cfg["nodes_csv"] == tmp_path / "nodes.csv"
    assert cfg["edges_csv"] == tmp_path / "edges.csv"
    assert "start_id" not in cfg


@pytest.mark.parametrize("env", [
    {"GNN_RAG_EPOCHS": "0"},
    {"GNN_RAG_EPOCHS": "10001"},
    {"GNN_RAG_SEED": "-1"},
])
def test_bad_training_options_fail_fast(env):
    with pytest.raises(ValueError):
        compose_train.training_options(env)
