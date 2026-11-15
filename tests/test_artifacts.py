import numpy as np

from gnn_rag.artifacts import load_artifacts, save_artifacts
from gnn_rag.split import split_edges
from gnn_rag.train import train


def test_roundtrip_keeps_node_ids_and_learned_vectors(graph, tmp_path):
    result = train(graph, split_edges(graph), epochs=2, hidden_dim=8, embed_dim=4)
    save_artifacts(tmp_path, graph, result)
    ids, vectors = load_artifacts(tmp_path)
    assert ids == graph.ids
    np.testing.assert_array_equal(vectors, result.embeddings)
