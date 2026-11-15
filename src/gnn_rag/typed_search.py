"""Type-filtered relation-conditioned ranking, exact and Voyager HNSW.

In this small demonstration, ranking all candidates exactly is the correctness
oracle. Voyager indexes target embeddings by type and is compared on the SAME
query/candidate sets; it is not assumed to improve retrieval quality.
"""
from __future__ import annotations

import numpy as np


class RelationRanker:
    def __init__(self, node_ids: tuple[str, ...], node_kinds: tuple[str, ...],
                 relation_names: tuple[str, ...], embeddings: np.ndarray,
                 relation_vectors: np.ndarray):
        self.ids = tuple(node_ids)
        self.kinds = tuple(node_kinds)
        self.relations = tuple(relation_names)
        self.id_to_row = {name: i for i, name in enumerate(node_ids)}
        self.rel_to_row = {name: i for i, name in enumerate(relation_names)}
        self.z = np.asarray(embeddings, dtype=np.float32)
        self.r = np.asarray(relation_vectors, dtype=np.float32)
        if self.z.shape != (len(self.ids), self.r.shape[1]) or self.r.shape[0] != len(self.relations):
            raise ValueError('Embeddings/relation vector dimensions mismatch')
        if len(set(self.ids)) != len(self.ids) or not np.isfinite(self.z).all() or not np.isfinite(self.r).all():
            raise ValueError('Non-finite/duplicate IDs in relation ranking artifacts')
        self._indices = {}

    def query_vector(self, head: int, relation_id: int) -> np.ndarray:
        q = self.z[int(head)] + self.r[int(relation_id)]
        return np.asarray(q / max(float(np.linalg.norm(q)), 1e-8), dtype=np.float32)

    def edge_score(self, head: int, relation_id: int, target: int) -> float:
        return float(np.dot(self.query_vector(head, relation_id), self.z[int(target)]))

    def exact(self, head: int, relation_id: int, candidates: np.ndarray,
              k: int | None = None) -> list[int]:
        candidates = np.asarray(candidates, dtype=np.int32)
        q = self.query_vector(head, relation_id)
        score = self.z[candidates] @ q
        ranked = sorted(zip(candidates.tolist(), score.tolist()), key=lambda item: (-item[1], item[0]))
        return [int(idx) for idx, _ in (ranked if k is None else ranked[:k])]

    def _index_for_kind(self, kind: str):
        if kind not in self._indices:
            from voyager import Index, Space
            positions = [i for i, k in enumerate(self.kinds) if k == kind]
            if not positions:
                raise ValueError(f'No nodes of kind {kind!r}')
            idx = Index(Space.Euclidean, num_dimensions=self.z.shape[1],
                        max_elements=len(positions), random_seed=42)
            id_map = {}
            for p in positions:
                id_map[int(idx.add_item(np.ascontiguousarray(self.z[p])))] = p
            self._indices[kind] = (idx, id_map)
        return self._indices[kind]

    def voyager(self, head: int, relation_id: int, candidates: np.ndarray, k: int) -> list[int]:
        """Relation-conditioned HNSW, filtered to the EXACT evaluation candidates.

        Grow the query to compensate for excluded training neighbors. Unlike
        an unrestricted latent nearest-neighbor walk, source/target types and
        relation are part of the lookup. This may scan all nodes for small KGs.
        """
        if k < 1:
            raise ValueError('k must be positive')
        candidates = np.asarray(candidates, dtype=np.int32)
        if not len(candidates):
            return []
        wanted = min(k, len(candidates))
        candidate_set = set(candidates.tolist())
        kinds = {self.kinds[int(i)] for i in candidates}
        if len(kinds) != 1:
            # For schema-less synthetic relations all node types are eligible.
            # A single typed HNSW index cannot handle this, so search per kind.
            found = []
            for kind in sorted(kinds):
                subset = np.asarray([i for i in candidates if self.kinds[int(i)] == kind])
                found.extend(self._query_kind(kind, head, relation_id, subset, wanted))
            return self.exact(head, relation_id, np.asarray(found, dtype=np.int32), wanted)
        return self._query_kind(next(iter(kinds)), head, relation_id, candidates, wanted)

    def _query_kind(self, kind: str, head: int, relation_id: int,
                    candidates: np.ndarray, k: int) -> list[int]:
        index, id_map = self._index_for_kind(kind)
        wanted = min(k, len(candidates))
        allowed = set(int(c) for c in candidates)
        count = min(len(id_map), max(wanted + 4, wanted * 2))
        q = self.query_vector(head, relation_id)
        while True:
            raw_ids, _ = index.query(np.ascontiguousarray(q), k=count)
            retrieved = [id_map[int(j)] for j in np.ravel(raw_ids) if int(j) in id_map and id_map[int(j)] in allowed]
            retrieved = list(dict.fromkeys(retrieved))
            if len(retrieved) >= wanted or count == len(id_map):
                # Re-rank HNSW candidates with the exact training score.
                return self.exact(head, relation_id, np.asarray(retrieved, dtype=np.int32), wanted)
            count = min(len(id_map), max(count + 1, count * 2))

    def known_edge_score(self, source_id: str, relation: str, target_id: str) -> float:
        return self.edge_score(self.id_to_row[source_id], self.rel_to_row[relation],
                               self.id_to_row[target_id])
