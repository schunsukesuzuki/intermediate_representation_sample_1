"""Small, auditable Polars boundary for a knowledge graph.

Node IDs are *external* strings; array indices are deterministic, contiguous integers.
No pandas, DGL or source-specific Excel paths are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

NODE_COLUMNS = {"node_id", "name", "topic", "kind", "description"}
EDGE_COLUMNS = {"source", "target", "weight"}


@dataclass(frozen=True)
class GraphData:
    nodes: pl.DataFrame
    edges: pl.DataFrame
    features: np.ndarray  # [num_nodes, num_topic + num_kind], float32
    senders: np.ndarray  # directed positive edges, int32
    receivers: np.ndarray
    weights: np.ndarray  # edge metadata (positive weights)
    ids: tuple[str, ...]
    id_to_index: dict[str, int]
    relation_ids: np.ndarray  # [num_triples]
    relation_names: tuple[str, ...]
    node_kinds: tuple[str, ...]

    @property
    def n_nodes(self) -> int:
        return len(self.ids)

    @property
    def positive_edges(self) -> set[tuple[int, int]]:
        return set(zip(self.senders.tolist(), self.receivers.tolist()))


def load_graph(nodes_path: Path | str, edges_path: Path | str) -> GraphData:
    nodes = pl.read_csv(nodes_path, infer_schema=False)
    edges = pl.read_csv(edges_path, infer_schema=False)
    if not NODE_COLUMNS.issubset(nodes.columns):
        raise ValueError(f"nodes.csv requires {sorted(NODE_COLUMNS)}")
    if not EDGE_COLUMNS.issubset(edges.columns):
        raise ValueError(f"edges.csv requires {sorted(EDGE_COLUMNS)}")
    if nodes.height < 6 or edges.height < 6:
        raise ValueError("At least 6 nodes and 6 edges are needed for a demonstrative split")
    if any(nodes[c].null_count() for c in NODE_COLUMNS):
        raise ValueError("Null node attributes are not supported")
    if any(edges[c].null_count() for c in EDGE_COLUMNS):
        raise ValueError("Null edge attributes are not supported")

    # Keep predictable IDs irrespective of original CSV row order.
    nodes = nodes.select(sorted(NODE_COLUMNS)).with_columns(pl.col("node_id").cast(pl.String))
    nodes = nodes.sort("node_id")
    ids = tuple(nodes.get_column("node_id").to_list())
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate node_id values")
    mapping = {key: pos for pos, key in enumerate(ids)}
    topics = sorted(nodes.get_column("topic").unique().to_list())
    kinds = sorted(nodes.get_column("kind").unique().to_list())
    topic_map = {v: i for i, v in enumerate(topics)}
    kind_map = {v: i + len(topics) for i, v in enumerate(kinds)}
    features = np.zeros((len(ids), len(topics) + len(kinds)), dtype=np.float32)
    for i, row in enumerate(nodes.iter_rows(named=True)):
        features[i, topic_map[row["topic"]]] = 1.0
        features[i, kind_map[row["kind"]]] = 1.0

    edges = edges.select("source", "target", "weight", *( ["relation"] if "relation" in edges.columns else [] )).with_columns(
        pl.col("source").cast(pl.String),
        pl.col("target").cast(pl.String),
        pl.col("weight").cast(pl.Float32, strict=True),
    ).sort("source", "target", maintain_order=True)
    src_ids = edges.get_column("source").to_list()
    dst_ids = edges.get_column("target").to_list()
    if any(s not in mapping or t not in mapping for s, t in zip(src_ids, dst_ids)):
        raise ValueError("An edge refers to a missing node")
    if any(s == t for s, t in zip(src_ids, dst_ids)):
        raise ValueError("Self-loops are not accepted in this demo")
    relations = (edges.get_column("relation").to_list() if "relation" in edges.columns
                 else ["related_to"] * len(src_ids))
    if len(set(zip(src_ids, relations, dst_ids))) != len(src_ids):
        raise ValueError("Duplicate (source, relation, target) triple")
    relation_names = tuple(sorted(set(relations)))
    relation_lookup = {r: i for i, r in enumerate(relation_names)}
    weight = edges.get_column("weight").to_numpy().astype(np.float32)
    if not np.all(np.isfinite(weight)) or np.any(weight <= 0):
        raise ValueError("Edge weights must be finite and positive")
    return GraphData(
        nodes=nodes, edges=edges, features=features,
        senders=np.array([mapping[s] for s in src_ids], dtype=np.int32),
        receivers=np.array([mapping[t] for t in dst_ids], dtype=np.int32),
        weights=weight, ids=ids, id_to_index=mapping,
        relation_ids=np.asarray([relation_lookup[r] for r in relations], dtype=np.int32),
        relation_names=relation_names,
        node_kinds=tuple(nodes.get_column("kind").to_list()),
    )
