"""Lightweight tests: small fixtures, no real biomedical data or Ollama required."""
from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from gnn_rag.data import load_graph
from gnn_rag.grounded import evidence_prompt, grounded_search, observed_training_edges
from gnn_rag.llm import OllamaClient
from gnn_rag.public_data import convert_hetionet
from gnn_rag.retrieval import VoyagerRetriever
from gnn_rag.split import split_edges


def test_grounded_context_edges_all_observed(graph):
    split = split_edges(graph, seed=4)
    observed = observed_training_edges(graph.edges, graph.ids, split.train)
    index = VoyagerRetriever(graph.ids, np.eye(graph.n_nodes, dtype=np.float32))
    path_list = grounded_search(index, observed, "n00", hops=2, top_k=3, beam_width=6)
    triples = {(r["source"], r["relation"], r["target"])
               for r in observed.iter_rows(named=True)}
    assert path_list
    for path in path_list:
        assert len(set(path.nodes)) == len(path.nodes)
        assert len(path.nodes) == len(path.edges) + 1
        for edge in path.edges:
            assert (edge.source, edge.relation, edge.target) in triples
    prompt = evidence_prompt(graph.nodes, path_list, "何が関係する？")
    assert "【観測済みエッジ" in prompt
    assert "何が関係する？" in prompt


def test_public_hetionet_converter_small_fixture(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    node_path = input_dir / "nodes.tsv"
    edge_path = input_dir / "edges.sif.gz"
    kinds = ["Disease", "Symptom", "Compound", "Gene"]
    with node_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "name", "kind"])
        for i in range(30):
            writer.writerow([f"id{i:02}", "Alzheimer's disease" if i == 0 else f"entity{i}", kinds[i % 4]])
    with gzip.open(edge_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source", "metaedge", "target"])
        for i in range(1, 25):
            writer.writerow(["id00", "DpS", f"id{i:02}"])
        for i in range(1, 24):
            writer.writerow([f"id{i:02}", "GiG", f"id{i+1:02}"])
    stats = convert_hetionet(node_path, edge_path, tmp_path / "out", max_nodes=18)
    assert 8 <= stats["edges"]
    graph = load_graph(tmp_path / "out/nodes.csv", tmp_path / "out/edges.csv")
    assert graph.n_nodes <= 18
    assert "relation" in graph.edges.columns
    assert "id00" in graph.ids


def test_ollama_json_and_local_only(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, *_): return b'{"message":{"content":"[E1] edge confirmed"}}'

    def stub(req, timeout):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data.decode())
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("gnn_rag.llm.urlopen", stub)
    content = OllamaClient("qwen3:0.6b").answer("Some evidence [E1]")
    assert content == "[E1] edge confirmed"
    assert captured["url"].endswith("/api/chat")
    assert captured["data"]["stream"] is False
    assert captured["data"]["think"] is False
    assert captured["data"]["messages"][1]["content"] == "Some evidence [E1]"
    with pytest.raises(ValueError):
        OllamaClient(endpoint="https://api.example.org").answer("question")
    with pytest.raises(ValueError):
        OllamaClient(endpoint="http://example.org:11434").answer("question")
    content = OllamaClient(endpoint="http://ollama:11434").answer("Some evidence [E1]")
    assert content == "[E1] edge confirmed"
    assert captured["url"] == "http://ollama:11434/api/chat"
