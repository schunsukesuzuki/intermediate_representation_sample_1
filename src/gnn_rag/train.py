"""Reproducible JAX training; no model weight reinitialization on inference."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .data import GraphData
from .model import binary_loss, edge_logits, encode, init_params
from .split import EdgeSplit


@dataclass(frozen=True)
class TrainingResult:
    params: dict
    embeddings: np.ndarray
    metrics: dict[str, float | int]


def auc_from_logits(positive: np.ndarray, negative: np.ndarray) -> float:
    """Exact ROC-AUC for two equally defined sets, without sklearn dependency."""
    if not len(positive) or not len(negative):
        raise ValueError("Both classes are required for ROC-AUC")
    comparison = positive[:, None] - negative[None, :]
    return float(np.mean(comparison > 0) + 0.5 * np.mean(comparison == 0))


def train(graph: GraphData, split: EdgeSplit, *, seed: int = 42,
          epochs: int = 60, learning_rate: float = 0.01,
          hidden_dim: int = 32, embed_dim: int = 16) -> TrainingResult:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs must be positive and learning_rate > 0")
    params = init_params(jax.random.PRNGKey(seed), graph.features.shape[1], hidden_dim, embed_dim)
    features = jnp.asarray(graph.features)
    train_edges = jnp.asarray(split.train)
    train_weights = jnp.asarray(split.train_weight)
    src, dst = train_edges[:, 0], train_edges[:, 1]
    val_pos, val_neg = jnp.asarray(split.validation), jnp.asarray(split.validation_neg)
    test_pos, test_neg = jnp.asarray(split.test), jnp.asarray(split.test_neg)

    def embed_fn(p: dict) -> jax.Array:
        return encode(p, features, src, dst, train_weights)

    def loss_fn(p: dict, negatives: jax.Array) -> jax.Array:
        z = embed_fn(p)
        pos_logits = edge_logits(p, z, train_edges)
        neg_logits = edge_logits(p, z, negatives)
        logits = jnp.concatenate([pos_logits, neg_logits])
        labels = jnp.concatenate([jnp.ones_like(pos_logits), jnp.zeros_like(neg_logits)])
        return binary_loss(logits, labels)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    embed_compiled = jax.jit(embed_fn)
    # Minimal Adam, explicit to keep the JAX demo dependency-light.
    zeros = jax.tree.map(jnp.zeros_like, params)
    m, v = zeros, zeros
    rng = np.random.default_rng(seed + 1)
    last_loss = float("nan")
    for step in range(1, epochs + 1):
        indices = rng.choice(len(split.train_neg_pool), size=len(split.train), replace=False)
        negatives = jnp.asarray(split.train_neg_pool[indices])
        loss, grad = value_and_grad(params, negatives)
        m = jax.tree.map(lambda previous, g: 0.9 * previous + 0.1 * g, m, grad)
        v = jax.tree.map(lambda previous, g: 0.999 * previous + 0.001 * g * g, v, grad)
        m_corr = jax.tree.map(lambda x: x / (1.0 - 0.9**step), m)
        v_corr = jax.tree.map(lambda x: x / (1.0 - 0.999**step), v)
        params = jax.tree.map(
            lambda p, mean, var: p - learning_rate * mean / (jnp.sqrt(var) + 1e-8),
            params, m_corr, v_corr,
        )
        last_loss = float(loss)

    embedded = embed_compiled(params)
    if not bool(jnp.all(jnp.isfinite(embedded))):
        raise ValueError("Non-finite learned embeddings")
    scores = {
        "validation_auc": auc_from_logits(
            np.asarray(edge_logits(params, embedded, val_pos)),
            np.asarray(edge_logits(params, embedded, val_neg)),
        ),
        "test_auc": auc_from_logits(
            np.asarray(edge_logits(params, embedded, test_pos)),
            np.asarray(edge_logits(params, embedded, test_neg)),
        ),
        "train_loss": last_loss,
        "epochs": epochs,
        "seed": seed,
        "num_train_positive_edges": len(split.train),
        "num_validation_positive_edges": len(split.validation),
        "num_test_positive_edges": len(split.test),
        "embedding_dim": embed_dim,
    }
    return TrainingResult(params=params, embeddings=np.asarray(embedded, dtype=np.float32), metrics=scores)
