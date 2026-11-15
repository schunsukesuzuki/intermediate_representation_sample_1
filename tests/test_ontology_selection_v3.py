"""No external files, models, Polars, or Voyager required."""
import json

import pytest

from gnn_rag.llm import OllamaClient
from gnn_rag.ontology import RELATIONS, relation_details


def test_all_hetionet_relationship_types_mapped():
    assert len(RELATIONS) == 24
    assert relation_details("DrD", "Disease", "Disease")["valid"]
    assert "文献抄録" in relation_details("DrD", "Disease", "Disease")["label"]
    assert relation_details("DlA", "Disease", "Anatomy")["valid"]
    assert not relation_details("DlA", "Disease", "Disease")["valid"]
    assert not relation_details("bogus", "Disease", "Disease")["valid"]
    assert relation_details("Gr>G", "Gene", "Gene")["directed"]


def test_select_evidence_json_schema_and_validated_ids(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, *_):
            return json.dumps({"message": {"content": json.dumps({
                "evidence_ids": ["E2", "E1"], "insufficient_evidence": False
            })}}).encode()

    def stub(req, timeout):
        captured["data"] = json.loads(req.data.decode())
        return Response()

    monkeypatch.setattr("gnn_rag.llm.urlopen", stub)
    answer = OllamaClient().select_evidence("任意の質問", ("E1", "E2"))
    assert answer.ids == ("E2", "E1")
    assert answer.status == "selected"
    assert set(captured["data"]["format"]["properties"]["evidence_ids"]["items"]["enum"]) == {"E1", "E2"}
    assert captured["data"]["think"] is False
    assert captured["data"]["stream"] is False


def test_id_fabrication_and_invalid_json_fail_closed(monkeypatch):
    def bad(_self, prompt, *, schema):
        return '{"evidence_ids":["E999"], "insufficient_evidence":false}'
    monkeypatch.setattr(OllamaClient, "_chat", bad)
    assert OllamaClient().select_evidence("?", ("E1",)).status == "unknown_ids"
    monkeypatch.setattr(OllamaClient, "_chat", lambda _self, prompt, *, schema: "bogus")
    assert OllamaClient().select_evidence("?", ("E1",)).status == "invalid_response"
    assert OllamaClient().select_evidence("?", ()).status == "insufficient"
