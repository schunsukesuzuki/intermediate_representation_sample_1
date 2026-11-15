"""JAX GraphSAGE-style weighted mean aggregator and link-prediction decoder.

Unlike the legacy prototype, all projection weights are trainable JAX PyTree
parameters. Train and inference share the same encode() function/parameters.
This is a *GraphSAGE-style* mean message passing network, not GAT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _linear(key: jax.Array, input_dim: int, output_dim: int) -> dict:
    # Fan-in scaling; deterministic given the supplied PRNG key.
    return {
        "w": jax.random.normal(key, (input_dim, output_dim), dtype=jnp.float32)
        * jnp.sqrt(2.0 / input_dim),
        "b": jnp.zeros((output_dim,), dtype=jnp.float32),
    }


def init_params(key: jax.Array, feature_dim: int, hidden_dim: int = 32,
                embed_dim: int = 16) -> dict:
    if min(feature_dim, hidden_dim, embed_dim) < 1:
        raise ValueError("All dimensions must be positive")
    k0, k1, k2 = jax.random.split(key, 3)
    return {
        "input": _linear(k0, feature_dim, hidden_dim),
        "sage_1": _linear(k1, hidden_dim * 2, hidden_dim),
        "sage_2": _linear(k2, hidden_dim * 2, embed_dim),
        "logit_scale": jnp.asarray(1.5, dtype=jnp.float32),
    }


def _weighted_neighbor_mean(h: jax.Array, senders: jax.Array,
                            receivers: jax.Array, weights: jax.Array) -> jax.Array:
    n = h.shape[0]
    # Directed edges are mirrored *only for message passing*, not for labels.
    sources = jnp.concatenate([senders, receivers])
    targets = jnp.concatenate([receivers, senders])
    w = jnp.concatenate([weights, weights])
    aggregate = jnp.zeros_like(h).at[targets].add(h[sources] * w[:, None])
    degree = jnp.zeros((n,), dtype=h.dtype).at[targets].add(w)
    return aggregate / jnp.maximum(degree[:, None], 1e-8)


def encode(params: dict, features: jax.Array, senders: jax.Array,
           receivers: jax.Array, weights: jax.Array) -> jax.Array:
    h = jax.nn.relu(features @ params["input"]["w"] + params["input"]["b"])
    for name in ("sage_1", "sage_2"):
        neighborhood = _weighted_neighbor_mean(h, senders, receivers, weights)
        h = jnp.concatenate([h, neighborhood], axis=-1)
        h = h @ params[name]["w"] + params[name]["b"]
        if name == "sage_1":
            h = jax.nn.relu(h)
    # Nonzero norms, even for isolated vertices, and cosine-like ANN vectors.
    return h / jnp.maximum(jnp.linalg.norm(h, axis=-1, keepdims=True), 1e-8)


def edge_logits(params: dict, embeddings: jax.Array, pairs: jax.Array) -> jax.Array:
    scale = jnp.exp(jnp.clip(params["logit_scale"], -2.0, 3.0))
    return scale * jnp.sum(embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]], axis=-1)


def binary_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    # Stable BCE-with-logits = softplus(x) - y*x.
    return jnp.mean(jax.nn.softplus(logits) - labels * logits)
