"""Save and reload the exact trained vectors (never initialized weights)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import GraphData
from .train import TrainingResult


def save_artifacts(directory: Path | str, graph: GraphData,
                   result: TrainingResult) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    graph.nodes.write_csv(directory / "nodes.csv")
    graph.edges.write_csv(directory / "edges.csv")
    np.save(directory / "embeddings.npy", result.embeddings, allow_pickle=False)
    # Safe numeric serialization (no pickle) for the fixed public demo architecture.
    np.savez(
        directory / "model_parameters.npz",
        input_w=np.asarray(result.params["input"]["w"]),
        input_b=np.asarray(result.params["input"]["b"]),
        sage1_w=np.asarray(result.params["sage_1"]["w"]),
        sage1_b=np.asarray(result.params["sage_1"]["b"]),
        sage2_w=np.asarray(result.params["sage_2"]["w"]),
        sage2_b=np.asarray(result.params["sage_2"]["b"]),
        logit_scale=np.asarray(result.params["logit_scale"]),
    )
    (directory / "metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (directory / "node_ids.json").write_text(
        json.dumps(graph.ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return directory


def load_artifacts(directory: Path | str) -> tuple[tuple[str, ...], np.ndarray]:
    directory = Path(directory)
    ids = tuple(json.loads((directory / "node_ids.json").read_text(encoding="utf-8")))
    embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)
    if len(ids) != len(embeddings):
        raise ValueError("node IDs and embeddings disagree")
    return ids, embeddings
