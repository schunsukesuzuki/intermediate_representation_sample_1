import pytest


@pytest.fixture
def graph(tmp_path):
    # Import lazily so config/HTTP-only tests do not require the full ML stack.
    from gnn_rag.data import load_graph
    from gnn_rag.synthetic import write_synthetic_dataset

    nodes, edges = write_synthetic_dataset(tmp_path)
    return load_graph(nodes, edges)
