"""Integration checks; requires Polars for KG table construction."""
from __future__ import annotations

import polars as pl

from gnn_rag.grounded import (EvidenceEdge, EvidencePath, evidence_catalog,
                              evidence_prompt, render_verified_answer)


def test_dr_d_and_dl_a_are_typed_and_cannot_become_pathology_claim():
    nodes = pl.DataFrame([
        {"node_id": "d1", "name": "Alzheimer's disease", "kind": "Disease"},
        {"node_id": "d2", "name": "Parkinson's disease", "kind": "Disease"},
        {"node_id": "a1", "name": "forebrain", "kind": "Anatomy"},
    ])
    p = EvidencePath(("d1", "d2", "a1"),
                     (EvidenceEdge("d1", "d2", "DrD"), EvidenceEdge("d2", "a1", "DlA")),
                     0.5)
    facts = evidence_catalog(nodes, [p])
    assert len(facts) == 2
    assert "疾患間の類似" in facts[0].statement
    assert "文献抄録" in facts[0].statement
    assert "Anatomy" in facts[1].statement
    prompt = evidence_prompt(nodes, [p], "どんな関係？")
    assert "DrD: 疾患間の類似" in prompt
    answer = render_verified_answer(facts, ["E1", "E999", "E2"])
    assert "E999" not in answer
    assert "forebrain（Anatomy）" in answer
    assert "共通の病理" not in answer


def test_unknown_or_mismatched_relation_is_not_evidence():
    nodes = pl.DataFrame([
        {"node_id": "d1", "name": "Disease1", "kind": "Disease"},
        {"node_id": "d2", "name": "Disease2", "kind": "Disease"},
    ])
    p = EvidencePath(("d1", "d2"), (EvidenceEdge("d1", "d2", "DlA"),), 0.5)
    assert evidence_catalog(nodes, [p]) == []
    assert "根拠IDがありません" in render_verified_answer([], ["E2"])
