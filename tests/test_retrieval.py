import numpy as np
import pytest

from gnn_rag.context import make_context
from gnn_rag.retrieval import VoyagerRetriever


def test_voyager_mapping_excludes_self_and_no_repeated_hop_nodes(graph):
    vec = np.eye(graph.n_nodes, dtype=np.float32)
    index = VoyagerRetriever(graph.ids, vec)
    neighbors = index.nearest("n00", top_k=3)
    assert len(neighbors) == 3
    assert "n00" not in [r.node_id for r in neighbors]
    paths = index.multi_hop("n00", hops=2, top_k=3, beam_width=4)
    assert 0 < len(paths) <= 4
    assert all(len(path.node_ids) == len(set(path.node_ids)) == 3 for path in paths)
    assert all(path.score >= 0 for path in paths)
    context = make_context(graph.nodes, paths, question="説明してください")
    assert "ANN" in context and "看護記録" in context and "説明してください" in context
    assert "LLM未接続" in context


def test_bad_id_and_bad_hops(graph):
    index = VoyagerRetriever(graph.ids, np.eye(graph.n_nodes, dtype=np.float32))
    with pytest.raises(KeyError):
        index.nearest("not-a-node")
    with pytest.raises(ValueError):
        index.multi_hop("n00", hops=graph.n_nodes)
