import jax
import jax.numpy as jnp
import numpy as np

from gnn_rag.model import edge_logits, encode, init_params
from gnn_rag.split import split_edges
from gnn_rag.train import auc_from_logits, train


def test_jax_embeddings_finite_and_parameter_gradients_exist(graph):
    split = split_edges(graph, seed=42)
    params = init_params(jax.random.PRNGKey(0), graph.features.shape[1], 12, 8)
    inputs = (jnp.asarray(graph.features), jnp.asarray(split.train[:, 0]),
              jnp.asarray(split.train[:, 1]), jnp.asarray(split.train_weight))

    def f(p):
        z = encode(p, *inputs)
        return jnp.sum(edge_logits(p, z, jnp.asarray(split.train)))

    gradients = jax.grad(f)(params)
    assert bool(jnp.all(jnp.isfinite(gradients["sage_1"]["w"])))
    assert float(jnp.linalg.norm(gradients["sage_1"]["w"])) > 0


def test_training_exports_learned_vectors(graph):
    split = split_edges(graph, seed=3)
    result = train(graph, split, epochs=3, seed=3, hidden_dim=12, embed_dim=8)
    assert result.embeddings.shape == (graph.n_nodes, 8)
    assert np.isfinite(result.embeddings).all()
    assert 0 <= result.metrics["test_auc"] <= 1
    params = init_params(jax.random.PRNGKey(3), graph.features.shape[1], 12, 8)
    initial = encode(params, jnp.asarray(graph.features), jnp.asarray(split.train[:, 0]),
                     jnp.asarray(split.train[:, 1]), jnp.asarray(split.train_weight))
    assert not np.allclose(result.embeddings, np.asarray(initial))


def test_auc_sanity():
    assert auc_from_logits(np.array([0.8, 0.9]), np.array([0.2, 0.3])) == 1.0
    assert auc_from_logits(np.array([0.2]), np.array([0.2])) == 0.5
