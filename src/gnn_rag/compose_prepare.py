"""Idempotent data preparation for `docker compose up`.

* synthetic: no I/O and no network required.
* hetionet: reuse consistent prepared CSV + metadata; otherwise validate/download
  official raw data and regenerate the bounded subgraph.

Never silently switch a requested Hetionet run to the synthetic dataset.
This job owns write access to /app/data; the training job mounts it read-only.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Mapping

from .fetch_hetionet import EDGES, NODES, ensure_hetionet_inputs, validate

DEFAULT_DATA_ROOT = Path("/app/data")


def _bool_setting(text: str) -> bool:
    value = text.strip().lower()
    if value in ("1", "yes", "on", "true"):
        return True
    if value in ("0", "no", "off", "false"):
        return False
    raise ValueError("HETIONET_AUTO_DOWNLOAD must be 1/0 (or true/false)")


def _read_csv(path: Path, columns: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not columns.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid CSV columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    if not rows or any(not all(row.get(col) for col in columns) for row in rows):
        raise ValueError(f"Empty or incomplete CSV rows in {path}")
    return rows


def cached_hetionet_is_usable(
    directory: Path, *, focus: str, max_nodes: int, seed: int,
) -> tuple[bool, str]:
    """Check v4 triple-preserving metadata and coherency, not just file existence.

    v3 caches collapsed edges to node pairs, so they MUST be resampled from
    the official raw data even when prior CSVs were correctly written.
    """
    nodes_file, edges_file = directory / "nodes.csv", directory / "edges.csv"
    meta_file = directory / "dataset_meta.json"
    if not all(path.is_file() for path in (nodes_file, edges_file, meta_file)):
        return False, "prepared nodes.csv / edges.csv / dataset_meta.json not all present"
    try:
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("dataset_meta.json must contain an object")
        if metadata.get("schema_version") != 4 or metadata.get("triple_preserving") is not True:
            return False, "old pair-collapsed cache; resample the v4 triple-preserving dataset"
        cached_focus = metadata.get("focus_name")
        if not isinstance(cached_focus, str) or cached_focus.casefold() != focus.casefold():
            return False, "requested focus differs from cached focus"
        if "requested_max_nodes" in metadata or "seed" in metadata:
            if metadata.get("requested_max_nodes") != max_nodes or metadata.get("seed") != seed:
                return False, "max_nodes or seed changed since preparation"
        elif (max_nodes, seed) != (120, 42):
            return False, "legacy cache has no parameters; non-default configuration requested"
        nodes = _read_csv(nodes_file, {"node_id", "name", "kind"})
        edges = _read_csv(edges_file, {"source", "target", "relation"})
        ids = [row["node_id"] for row in nodes]
        id_set = set(ids)
        if len(ids) != len(id_set):
            return False, "duplicate node identifiers"
        if not (12 <= len(ids) <= max_nodes) or len(edges) < 8:
            return False, "node/edge count outside bounded Hetionet demo limits"
        if metadata.get("nodes") != len(ids) or metadata.get("edges") != len(edges):
            return False, "CSV counts do not match dataset_meta.json"
        if metadata.get("focus_id") not in id_set:
            return False, "focus node is not in the prepared subgraph"
        if any(row["source"] not in id_set or row["target"] not in id_set
               for row in edges):
            return False, "edge refers to missing node"
        triples = [(row['source'], row['relation'], row['target']) for row in edges]
        if len(set(triples)) != len(triples):
            return False, "duplicate typed triples in prepared CSV"
    except (OSError, UnicodeError, ValueError, TypeError, csv.Error) as exc:
        return False, f"bad prepared data: {exc}"
    return True, f"{len(nodes)} nodes / {len(edges)} edges; focus={focus}"



def cached_scaled_hetionet_is_usable(directory: Path, *, target: int, max_nodes: int, seed: int,
                                    focus: str | None = None) -> tuple[bool, str]:
    """Check metadata and full sampled CSV shape before reusing a costly cache."""
    meta_path = directory / 'dataset_meta.json'
    if not meta_path.is_file():
        return False, 'v6 metadata missing'
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if (meta.get('schema_version') != 6 or meta.get('sampling_mode') != 'full_hetionet_structural_sample'
            or meta.get('requested_target_triples') != target
            or meta.get('requested_max_nodes') != max_nodes or meta.get('seed') != seed
            or (focus is not None and meta.get('requested_focus') != focus)):
            return False, 'sample parameters or schema changed'
        nodes = _read_csv(directory / 'nodes.csv', {'node_id', 'name', 'kind'})
        edges = _read_csv(directory / 'edges.csv', {'source', 'target', 'relation'})
        ids = {row['node_id'] for row in nodes}
        if (len(ids) != len(nodes) or len(nodes) > max_nodes or len(edges) != target
            or meta.get('nodes') != len(nodes) or meta.get('edges') != len(edges)):
            return False, 'sample counts do not match'
        triples = {(e['source'], e['relation'], e['target']) for e in edges}
        if len(triples) != len(edges) or any(e['source'] not in ids or e['target'] not in ids for e in edges):
            return False, 'duplicate or orphan triple'
        return True, f"{len(nodes)} nodes / {len(edges)} typed triples"
    except (OSError, ValueError, TypeError, csv.Error, KeyError, UnicodeError) as exc:
        return False, f'invalid CSV cache: {exc}'

def _cached_raw_files(raw_dir: Path) -> tuple[Path, Path]:
    """No-download mode validates both files; does not fall back to the network."""
    nodes, edges = raw_dir / NODES.filename, raw_dir / EDGES.filename
    for path, artifact in ((nodes, NODES), (edges, EDGES)):
        validate(path, artifact)
    return nodes, edges


def prepare_data(
    environ: Mapping[str, str] | None = None, *, data_root: Path = DEFAULT_DATA_ROOT,
) -> dict:
    env = os.environ if environ is None else environ
    dataset = env.get("GNN_RAG_DATASET", "synthetic").strip().lower()
    if dataset not in ("synthetic", "hetionet"):
        raise ValueError("GNN_RAG_DATASET must be synthetic or hetionet")
    if dataset == "synthetic":
        print("[prepare-data] synthetic: no download or CSV conversion needed", flush=True)
        return {"dataset": dataset, "status": "not_required"}

    focus = env.get("HETIONET_FOCUS", "Alzheimer's disease").strip()
    max_nodes = int(env.get("HETIONET_MAX_NODES", "120"))
    seed = int(env.get("GNN_RAG_SEED", "42"))
    target_triples = int(env.get("HETIONET_TARGET_TRIPLES", "0"))
    allow_download = _bool_setting(env.get("HETIONET_AUTO_DOWNLOAD", "1"))
    if not focus:
        raise ValueError("HETIONET_FOCUS must not be empty")
    if target_triples > 0:
        if target_triples < 100 or not 50 <= max_nodes <= 47031:
            raise ValueError('Full sampler requires target_triples >=100 and max_nodes 50..47031')
    elif not 12 <= max_nodes <= 600:
        raise ValueError("HETIONET_MAX_NODES must be between 12 and 600 in legacy mode")
    if seed < 0:
        raise ValueError("GNN_RAG_SEED must be nonnegative")

    output = data_root / "hetionet_demo"
    check = (lambda: cached_scaled_hetionet_is_usable(output, target=target_triples,
                  max_nodes=max_nodes, seed=seed, focus=focus)) if target_triples > 0 else (lambda: cached_hetionet_is_usable(
                  output, focus=focus, max_nodes=max_nodes, seed=seed))
    ready, reason = check()
    if ready:
        print(f"[prepare-data] using prepared Hetionet cache: {reason}", flush=True)
        return {"dataset": dataset, "status": "cached", "details": reason}
    print(f"[prepare-data] preparing Hetionet: {reason}", flush=True)
    raw_dir = data_root / "raw"
    if allow_download:
        nodes, edges = ensure_hetionet_inputs(raw_dir, download=True)
    else:
        nodes, edges = _cached_raw_files(raw_dir)

    # Lazy import: the synthetic path needs neither Polars nor dataset download.
    if target_triples > 0:
        from .hetionet_scale import convert_full_hetionet
        result = convert_full_hetionet(nodes, edges, output,
                   target_triples=target_triples, max_nodes=max_nodes, seed=seed, focus=focus)
    else:
        from .public_data import convert_hetionet
        result = convert_hetionet(nodes, edges, output, focus=focus,
                                  max_nodes=max_nodes, seed=seed)
    ready, reason = check()
    if not ready:
        raise RuntimeError(f"Hetionet conversion finished but CSV validation failed: {reason}")
    print(f"[prepare-data] ready: {reason}", flush=True)
    return {"dataset": dataset, "status": "created", "metadata": result}


def main() -> None:
    info = prepare_data()
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
