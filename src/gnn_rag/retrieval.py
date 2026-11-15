"""Voyager (HNSW) ANN plus bounded multi-hop latent-space exploration.

A chain of ANN hits is *not* necessarily a chain of KG edges. Paths here are
semantic search traces rather than verified causal/graph paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from voyager import Index, Space


@dataclass(frozen=True)
class Neighbor:
    node_id: str
    distance: float


@dataclass(frozen=True)
class SearchPath:
    node_ids: tuple[str, ...]
    distances: tuple[float, ...]

    @property
    def score(self) -> float:
        return float(sum(self.distances))


class VoyagerRetriever:
    def __init__(self, node_ids: tuple[str, ...], embeddings: np.ndarray):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(node_ids) or not embeddings.shape[1]:
            raise ValueError("embeddings must be [len(node_ids), dimension]")
        if len(node_ids) < 2 or len(node_ids) != len(set(node_ids)):
            raise ValueError("node IDs must be unique and contain at least 2 entries")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must be finite")
        self.node_ids = node_ids
        self.node_to_row = {node_id: row for row, node_id in enumerate(node_ids)}
        self.embeddings = np.ascontiguousarray(embeddings)
        self.index = Index(Space.Euclidean, num_dimensions=embeddings.shape[1],
                           max_elements=len(node_ids), random_seed=1)
        # Explicit map: Voyager item IDs are not guaranteed to equal node IDs.
        self.voyager_to_node: dict[int, str] = {}
        for node_id, vector in zip(node_ids, embeddings):
            item_id = int(self.index.add_item(vector))
            self.voyager_to_node[item_id] = node_id

    def nearest(self, node_id: str, *, top_k: int = 4,
                exclude: frozenset[str] = frozenset()) -> list[Neighbor]:
        if node_id not in self.node_to_row:
            raise KeyError(f"Unknown node_id: {node_id}")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        vector = self.embeddings[self.node_to_row[node_id]]
        # Grow k only when visited nodes remove too many candidates. Requesting
        # k = num_nodes on every hop would destroy HNSW's latency advantage.
        requested = min(len(self.node_ids), max(top_k + len(exclude) + 1, 4))
        while True:
            ids, distances = self.index.query(vector, k=requested)
            found = [
                Neighbor(self.voyager_to_node[int(i)], float(dist))
                for i, dist in zip(np.ravel(ids), np.ravel(distances))
                if self.voyager_to_node[int(i)] != node_id
                and self.voyager_to_node[int(i)] not in exclude
            ]
            found = sorted(found, key=lambda n: (n.distance, n.node_id))
            if len(found) >= top_k or requested == len(self.node_ids):
                return found[:top_k]
            requested = min(len(self.node_ids), requested * 2)

    def multi_hop(self, start_node_id: str, *, hops: int = 3, top_k: int = 3,
                  beam_width: int = 8) -> list[SearchPath]:
        if start_node_id not in self.node_to_row:
            raise KeyError(f"Unknown node_id: {start_node_id}")
        if hops < 1 or top_k < 1 or beam_width < 1:
            raise ValueError("hops, top_k, beam_width must be >= 1")
        if hops >= len(self.node_ids):
            raise ValueError("hops must be smaller than node count (no revisits)")
        frontier = [SearchPath((start_node_id,), ())]
        for _ in range(hops):
            expanded: list[SearchPath] = []
            for path in frontier:
                for neighbor in self.nearest(
                    path.node_ids[-1], top_k=top_k, exclude=frozenset(path.node_ids),
                ):
                    expanded.append(SearchPath(
                        (*path.node_ids, neighbor.node_id),
                        (*path.distances, neighbor.distance),
                    ))
            # Prevent O(top_k**hops) materialization; keep best semantic traces.
            frontier = sorted(expanded, key=lambda p: (p.score, p.node_ids))[:beam_width]
            if not frontier:
                break
        return frontier
