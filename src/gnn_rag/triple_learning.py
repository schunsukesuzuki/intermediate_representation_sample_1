"""Relation-conditioned JAX link prediction over (head, relation, tail) triples.

Model: GraphSAGE encoder + translational query vector normalize(z_head + r).
Target scores: dot(query, z_tail); higher is better. This is a small illustrative
relation decoder (not a claim to reproduce TransE or a production KGE model).
Train negatives use TRAIN triples only and public target-type constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from .model import encode, init_params
from .ontology import RELATIONS

if TYPE_CHECKING:
    from .data import GraphData


@dataclass(frozen=True)
class TripleSplit:
    train: np.ndarray  # [N, 3] integer head/relation/tail
    validation: np.ndarray
    test: np.ndarray
    train_weights: np.ndarray
    grouped_pairs: int


def triples_from_graph(graph: 'GraphData') -> np.ndarray:
    return np.column_stack((graph.senders, graph.relation_ids, graph.receivers)).astype(np.int32)


def triple_split(graph: 'GraphData', seed: int = 42, valid_fraction: float = .15,
                 test_fraction: float = .15) -> TripleSplit:
    """Keep all triples for each unordered node pair in ONE partition.

    Reserve >=2 training groups per relation when possible. This avoids direct
    train-val/test topological leakage and prevents rare relations disappearing
    entirely from the training vocabulary. Not a temporal/inductive split.
    """
    if not (0 < valid_fraction < .4 and 0 < test_fraction < .4):
        raise ValueError('Invalid split fractions')
    triples = triples_from_graph(graph)
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (s, _, t) in enumerate(triples):
        groups.setdefault(tuple(sorted((int(s), int(t)))), []).append(i)
    rng = np.random.default_rng(seed)
    group_items = list(groups.values())
    rng.shuffle(group_items)
    total = len(group_items)
    if total < 8:
        raise ValueError('At least 8 distinct node-pair groups required')
    relation_group_count = np.bincount(
        [int(r) for grp in group_items for r in set(triples[grp, 1])],
        minlength=len(graph.relation_names),
    )
    targets = {'test': max(1, round(total * test_fraction)),
               'validation': max(1, round(total * valid_fraction))}
    selected = {'test': [], 'validation': [], 'train': []}
    for group in group_items:
        relations = set(int(v) for v in triples[group, 1])
        can_withhold = all(relation_group_count[r] >= 3 for r in relations)
        dest = next((name for name in ('test', 'validation')
                     if can_withhold and len(selected[name]) < targets[name]), 'train')
        selected[dest].extend(group)
        if dest != 'train':
            for r in relations:
                relation_group_count[r] -= 1
    if not all(selected[name] for name in ('train', 'test', 'validation')):
        raise ValueError('Insufficient supported pairs for train/validation/test; enlarge the KG')
    def extract(name: str) -> np.ndarray:
        return triples[np.asarray(selected[name], dtype=np.int32)].copy()
    return TripleSplit(extract('train'), extract('validation'), extract('test'),
                       graph.weights[np.asarray(selected['train'], dtype=np.int32)].copy(), total)


def relation_target_kind(graph: 'GraphData', relation_id: int) -> str | None:
    name = graph.relation_names[int(relation_id)]
    if name in RELATIONS:
        return RELATIONS[name].target_kind
    # A generic relation has no declared ontology: do not peek at withheld labels.
    return None


def typed_candidate_ids(graph: 'GraphData', head: int, relation_id: int) -> np.ndarray:
    target_kind = relation_target_kind(graph, relation_id)
    return np.asarray([i for i, kind in enumerate(graph.node_kinds)
                       if i != int(head) and (target_kind is None or kind == target_kind)],
                      dtype=np.int32)


def candidate_without_observed(graph: 'GraphData', head: int, relation_id: int,
                               known: np.ndarray) -> np.ndarray:
    """Known = TRAIN ONLY for fitting; training edges excluded from ranking."""
    relation = graph.relation_names[relation_id]
    spec = RELATIONS.get(relation)
    symmetric = bool(spec is not None and not spec.directed and
                     spec.source_kind == spec.target_kind)
    blocked = set()
    for h, r, t in np.asarray(known).reshape(-1, 3):
        if int(r) != relation_id:
            continue
        if int(h) == head:
            blocked.add(int(t))
        if symmetric and int(t) == head:
            blocked.add(int(h))
    return np.asarray([i for i in typed_candidate_ids(graph, head, relation_id)
                       if i not in blocked], dtype=np.int32)


def sample_typed_negatives(graph: 'GraphData', positives: np.ndarray,
                           known_train: np.ndarray, rng: np.random.Generator,
                           *, per_positive: int = 1) -> np.ndarray:
    """No test/validation positive labels are consulted at training time.

    Unobserved doesn't necessarily mean false. Sampling only candidates of the
    declared relation target type prevents trivial cross-type negatives.
    """
    available = {}
    results = []
    for h, r, _ in np.asarray(positives).reshape(-1, 3):
        key = (int(h), int(r))
        if key not in available:
            available[key] = candidate_without_observed(graph, *key, known_train)
        candidates = available[key]
        if not len(candidates):
            continue
        picks = rng.choice(candidates, size=per_positive, replace=True)
        results.extend([(key[0], key[1], int(t)) for t in picks])
    return np.asarray(results, dtype=np.int32).reshape(-1, 3)


def sample_ranking_batch(graph: 'GraphData', positives: np.ndarray,
                         known_train: np.ndarray, rng: np.random.Generator,
                         n_negatives: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Sampled-softmax batch: one known positive vs same-(head,relation,type) tails.

    Negatives are drawn only after checking TRAIN triples. Incomplete KGs can
    contain unknown positives in the negative sample, a documented limitation.
    """
    if n_negatives < 1:
        raise ValueError('n_negatives must be >=1')
    valid_positive = []
    all_negative = []
    pool = {}
    for h, r, t in np.asarray(positives).reshape(-1, 3):
        key = (int(h), int(r))
        if key not in pool:
            pool[key] = candidate_without_observed(graph, *key, known_train)
        if not len(pool[key]):
            continue
        valid_positive.append((int(h), int(r), int(t)))
        all_negative.append([(key[0], key[1], int(x))
            for x in rng.choice(pool[key], size=n_negatives, replace=True)])
    return (np.asarray(valid_positive, dtype=np.int32).reshape(-1, 3),
            np.asarray(all_negative, dtype=np.int32).reshape(-1, n_negatives, 3))



class CachedTypedNegativeSampler:
    """Indexed same-relation/same-tail-type negative sampler for larger graphs.

    Uses TRAIN triples exclusively. Candidate sets are materialized lazily per
    (head, relation) and reused across epochs. No node-id/clan information is
    added to the model input. It does not claim sampled non-edges are true negatives.
    """
    def __init__(self, graph: 'GraphData', train: np.ndarray):
        from collections import defaultdict
        self.graph = graph
        self.blocked = defaultdict(set)
        self.candidates = {}
        self.kind_nodes = {}
        for i, kind in enumerate(graph.node_kinds):
            self.kind_nodes.setdefault(kind, []).append(i)
        for h, r, t in np.asarray(train, dtype=np.int32).reshape(-1, 3):
            self.blocked[(int(h), int(r))].add(int(t))
        self.all_nodes = np.arange(graph.n_nodes, dtype=np.int32)

    def for_query(self, h: int, r: int) -> np.ndarray:
        key = (int(h), int(r))
        if key not in self.candidates:
            kind = relation_target_kind(self.graph, int(r))
            options = self.all_nodes if kind is None else self.kind_nodes.get(kind, [])
            excluded = self.blocked[key] | {int(h)}
            self.candidates[key] = np.asarray(
                [int(x) for x in options if int(x) not in excluded], dtype=np.int32)
        return self.candidates[key]

    def draw(self, positives: np.ndarray, rng: np.random.Generator,
             n_negatives: int) -> tuple[np.ndarray, np.ndarray]:
        pos = []
        neg = []
        for h, r, t in np.asarray(positives, dtype=np.int32).reshape(-1, 3):
            pool = self.for_query(int(h), int(r))
            if not len(pool):
                continue
            picked = rng.choice(pool, size=n_negatives, replace=True)
            pos.append((int(h), int(r), int(t)))
            neg.append([(int(h), int(r), int(x)) for x in picked])
        return (np.asarray(pos, dtype=np.int32).reshape(-1, 3),
                np.asarray(neg, dtype=np.int32).reshape(-1, n_negatives, 3))


def expand_symmetric_training(graph: 'GraphData', train: np.ndarray) -> np.ndarray:
    """Treat same-kind, non-directed Hetionet relation links as bidirectional.

    This augmentation is TRAIN ONLY. Directed Gr>G and cross-kind relations
    are never mirrored. All original triples remain in original orientation.
    """
    triples = {tuple(map(int, row)) for row in train}
    for h, r, t in np.asarray(train).reshape(-1, 3):
        spec = RELATIONS.get(graph.relation_names[int(r)])
        if spec is not None and not spec.directed and spec.source_kind == spec.target_kind:
            triples.add((int(t), int(r), int(h)))
    return np.asarray(sorted(triples), dtype=np.int32).reshape(-1, 3)


def name_hash_features(graph: 'GraphData', buckets: int = 64, *,
                       feature_mode: str = 'kind_name') -> np.ndarray:
    """Fixed pre-split lexical name features; no additional model or corpus.

    Avoid duplicate topic=kind columns in the Hetionet adapter, and normalize
    character n-gram features to avoid favoring long names.
    """
    if feature_mode not in ('kind_name', 'kind_only'):
        raise ValueError('feature_mode must be kind_name or kind_only')
    if feature_mode == 'kind_name' and buckets < 8:
        raise ValueError('At least 8 hash buckets required')
    kinds = sorted(set(graph.node_kinds))
    if feature_mode == 'kind_only':
        # Strong ablation: no name, node identity, cluster ID, or link label
        # enters the feature matrix; same-kind nodes have identical features.
        kind_map = {v: i for i, v in enumerate(kinds)}
        features = np.zeros((graph.n_nodes, len(kinds)), dtype=np.float32)
        for i, kind in enumerate(graph.node_kinds):
            features[i, kind_map[kind]] = 1.0
        return features
    topics = graph.nodes.get_column('topic').to_list()
    use_topics = any(str(a) != str(b) for a, b in zip(topics, graph.node_kinds))
    topic_set = sorted(set(topics)) if use_topics else []
    result = np.zeros((graph.n_nodes, len(kinds) + len(topic_set) + buckets), dtype=np.float32)
    kind_map = {v: i for i, v in enumerate(kinds)}
    topic_map = {v: i + len(kinds) for i, v in enumerate(topic_set)}
    offset = len(kinds) + len(topic_set)
    names = graph.nodes.get_column('name').to_list()
    for i, name in enumerate(names):
        result[i, kind_map[graph.node_kinds[i]]] = 1
        if use_topics:
            result[i, topic_map[topics[i]]] = 1
        name = f'^{str(name).casefold()}$'
        for j in range(max(0, len(name) - 2)):
            feature = name[j:j+3]
            bucket = int.from_bytes(hashlib.sha256(feature.encode('utf8')).digest()[:4], 'big') % buckets
            result[i, offset + bucket] += 1
        text_norm = np.linalg.norm(result[i, offset:])
        if text_norm:
            result[i, offset:] /= text_norm
    return result


def init_triple_params(key: jax.Array, feature_dim: int, n_relations: int,
                       hidden_dim: int = 32, embed_dim: int = 16) -> dict:
    if n_relations < 1:
        raise ValueError('Expected at least one relation')
    key_gnn, key_rel = jax.random.split(key)
    p = init_params(key_gnn, feature_dim, hidden_dim, embed_dim)
    p['relation_vectors'] = jax.random.normal(key_rel, (n_relations, embed_dim)) * .15
    return p


def relation_queries(z: jax.Array, relation_vectors: jax.Array,
                     pairs: jax.Array) -> jax.Array:
    q = z[pairs[:, 0]] + relation_vectors[pairs[:, 1]]
    return q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-8)


def triple_logits(params: dict, z: jax.Array, triples: jax.Array) -> jax.Array:
    q = relation_queries(z, params['relation_vectors'], triples)
    scale = jnp.exp(jnp.clip(params['logit_scale'], -2., 3.))
    return scale * jnp.sum(q * z[triples[:, 2]], axis=-1)


@dataclass(frozen=True)
class TripleTraining:
    params: dict
    embeddings: np.ndarray
    features: np.ndarray
    metrics: dict


def train_triples(graph: 'GraphData', split: TripleSplit, *, seed: int = 42,
                  epochs: int = 25, learning_rate: float = .01,
                  hidden_dim: int = 32, embed_dim: int = 16,
                  negatives_per_positive: int = 4,
                  feature_mode: str = 'kind_name',
                  max_positives_per_epoch: int | None = None) -> TripleTraining:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError('Invalid train hyperparameters')
    features_np = name_hash_features(graph, feature_mode=feature_mode)
    features = jnp.asarray(features_np)
    augmented_train = expand_symmetric_training(graph, split.train)
    train_triple_jax = jnp.asarray(augmented_train)
    # Aggregate once per unordered pair: multiple relations between the same
    # nodes must not accidentally multiply their GraphSAGE message weight.
    adjacency_weights = {}
    for (h, _, t), weight in zip(split.train, split.train_weights):
        key = tuple(sorted((int(h), int(t))))
        adjacency_weights.setdefault(key, []).append(float(weight))
    unique_pairs = np.asarray(sorted(adjacency_weights), dtype=np.int32)
    weights = jnp.asarray([np.mean(adjacency_weights[tuple(p)]) for p in unique_pairs],
                          dtype=jnp.float32)
    adjacency = jnp.asarray(unique_pairs)
    params = init_triple_params(jax.random.PRNGKey(seed), features.shape[1],
                                len(graph.relation_names), hidden_dim, embed_dim)
    def get_embeddings(p):
        return encode(p, features, adjacency[:, 0],
                      adjacency[:, 1], weights)
    @jax.jit
    def update_loss(p, positives, negatives):
        z = get_embeddings(p)
        pos = triple_logits(p, z, positives)
        neg = triple_logits(p, z, negatives.reshape(-1, 3)).reshape(positives.shape[0], -1)
        # Sampled multiclass ranking loss, aligned with Top-K retrieval.
        logits = jnp.concatenate((pos[:, None], neg), axis=1)
        return jnp.mean(jax.nn.logsumexp(logits, axis=1) - pos)
    grad_fn = jax.jit(jax.value_and_grad(update_loss))
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    rng = np.random.default_rng(seed + 1)
    last = None
    cached_sampler = CachedTypedNegativeSampler(graph, augmented_train) if len(augmented_train) >= 500 else None
    eligible_train = augmented_train
    if cached_sampler is not None:
        eligible_train = np.asarray([row for row in augmented_train
             if len(cached_sampler.for_query(int(row[0]), int(row[1])))], dtype=np.int32).reshape(-1, 3)
        if not len(eligible_train):
            raise ValueError('All training triples have exhausted same-type negatives')
    if max_positives_per_epoch is not None and max_positives_per_epoch < 1:
        raise ValueError('max_positives_per_epoch must be positive')
    for epoch in range(1, epochs+1):
        epoch_train = eligible_train
        if max_positives_per_epoch and len(eligible_train) > max_positives_per_epoch:
            inds = rng.choice(len(eligible_train), size=max_positives_per_epoch, replace=False)
            epoch_train = eligible_train[inds]
        if cached_sampler is None:
            positives, negatives = sample_ranking_batch(
                graph, epoch_train, augmented_train, rng,
                n_negatives=negatives_per_positive)
        else:
            positives, negatives = cached_sampler.draw(epoch_train, rng, negatives_per_positive)
        if len(positives) == 0:
            raise ValueError('No typed negative candidates: increase subgraph size or modify split')
        loss, grad = grad_fn(params, jnp.asarray(positives), jnp.asarray(negatives))
        m = jax.tree.map(lambda a, b: .9*a + .1*b, m, grad)
        v = jax.tree.map(lambda a, b: .999*a + .001*b*b, v, grad)
        mm = jax.tree.map(lambda x: x/(1-.9**epoch), m)
        vv = jax.tree.map(lambda x: x/(1-.999**epoch), v)
        params = jax.tree.map(lambda p, a, b: p-learning_rate*a/(jnp.sqrt(b)+1e-8),
                              params, mm, vv)
        last = float(loss)
    z = np.asarray(get_embeddings(params), dtype=np.float32)
    if not np.all(np.isfinite(z)):
        raise ValueError('Non-finite triple embeddings')
    metrics = {'train_loss': last, 'epochs': epochs, 'seed': seed,
               'train_triples': len(split.train),
               'train_positives_after_symmetric_augmentation': len(augmented_train),
               'validation_triples': len(split.validation),
               'test_triples': len(split.test), 'embedding_dim': embed_dim,
               'n_relations': len(graph.relation_names),
               'train_adjacency_pairs': len(unique_pairs),
               'max_ranking_positives_per_epoch': max_positives_per_epoch,
               'ranking_eligible_training_positives': len(eligible_train),
               'feature_dim': features.shape[1], 'feature_mode': feature_mode,
               'negative_sampling': 'same target node type; exclude training positive triples',
               'negative_samples_per_positive': negatives_per_positive,
               'training_objective': 'sampled softmax ranking (one positive, N typed negatives)',
               'model': 'GraphSAGE encoder + relation-translated cosine query'}
    return TripleTraining(params, z, features_np, metrics)
