"""Convert a bounded, source-linked Hetionet subgraph to the existing CSV schema.

Source files must be obtained by the user from hetio/hetionet. No downloaded
biomedical corpus is redistributed with this repository. SIF exports do not
carry the licensing / original source attributes of JSON exports.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from collections import defaultdict
from pathlib import Path

import polars as pl

# Prefer concept types that make compact, human-readable multi-hop explanations.
KIND_ORDER = (
    "Disease", "Symptom", "Compound", "Anatomy", "Gene", "Pathway",
    "Side Effect", "Pharmacologic Class", "Biological Process",
    "Molecular Function", "Cellular Component",
)


def _edge_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) != 3:
                raise ValueError("Expected 3-column Hetionet SIF: source, metaedge, target")
            source, relation, target = (str(x).strip() for x in row)
            if (source.lower(), relation.lower(), target.lower()) == (
                "source", "metaedge", "target",
            ):
                continue
            if not source or not target or not relation:
                raise ValueError("Malformed SIF edge")
            yield source, relation, target


def _round_robin(candidates: set[str], attributes: dict[str, dict], limit: int, seed: int):
    groups: dict[str, list[str]] = defaultdict(list)
    for node in candidates:
        if node in attributes:
            groups[attributes[node]["kind"]].append(node)
    for group in groups.values():
        group.sort(key=lambda node: hashlib.sha1(f"{seed}|{node}".encode()).hexdigest())
    selected: list[str] = []
    order = [kind for kind in KIND_ORDER if kind in groups] + sorted(set(groups) - set(KIND_ORDER))
    while len(selected) < limit and any(groups.values()):
        for kind in order:
            if groups[kind] and len(selected) < limit:
                selected.append(groups[kind].pop())
    return selected


def convert_hetionet(
    nodes_tsv: Path, edges_sif_gz: Path, output: Path,
    *, focus: str = "Alzheimer's disease", max_nodes: int = 120, seed: int = 42,
) -> dict:
    """Sample a bounded two-hop neighborhood, retaining distinct typed triples.

    Bounded size is intentional: the original O(n^2) negative enumeration in
    split_edges() is unsuitable for the full 47k-node Hetionet graph.
    """
    if max_nodes < 12 or max_nodes > 600:
        raise ValueError("max_nodes must be 12..600 for this explanatory implementation")
    if not nodes_tsv.is_file() or not edges_sif_gz.is_file():
        missing = [str(p) for p in (nodes_tsv, edges_sif_gz) if not p.is_file()]
        raise FileNotFoundError(
            "Missing Hetionet input(s): " + ", ".join(missing) +
            ". With Docker Compose, use "
            "`docker compose --profile hetionet run --build --rm prepare-hetionet` "
            "to download/verify the official dataset automatically."
        )
    output.mkdir(parents=True, exist_ok=True)
    with open(nodes_tsv, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"id", "name", "kind"}.issubset(reader.fieldnames or []):
            raise ValueError("Hetionet nodes TSV needs columns id, name, kind")
        attrs = {r["id"]: {"name": r["name"], "kind": r["kind"]}
                 for r in reader if r.get("id")}
    matches = [key for key, row in attrs.items() if row["name"].casefold() == focus.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact focus node named {focus!r}; found {len(matches)}")
    focal = matches[0]

    first_neighbors: set[str] = set()
    for s, _, t in _edge_rows(edges_sif_gz):
        if s == focal and t in attrs:
            first_neighbors.add(t)
        if t == focal and s in attrs:
            first_neighbors.add(s)
    if not first_neighbors:
        raise ValueError("Focus node has no SIF neighbors")

    # Reserve space for second-hop concepts to get more than a star graph.
    first_budget = min(max_nodes - 1, max(8, int(max_nodes * 0.60)))
    first = _round_robin(first_neighbors, attrs, first_budget, seed)
    selected = set([focal, *first])
    second_neighbors: set[str] = set()
    for s, _, t in _edge_rows(edges_sif_gz):
        if s in selected and t in attrs and t not in selected:
            second_neighbors.add(t)
        if t in selected and s in attrs and s not in selected:
            second_neighbors.add(s)
    selected.update(_round_robin(second_neighbors, attrs, max_nodes - len(selected), seed + 1))

    # v4: preserve all distinct directed (source, relation, target) triples.
    # Do NOT collapse relations for the same unordered node pair.
    triples: set[tuple[str, str, str]] = set()
    for source, relation, target in _edge_rows(edges_sif_gz):
        if source in selected and target in selected and source != target:
            triples.add((source, relation, target))
    if len(triples) < 8:
        raise ValueError("Sampled subgraph has fewer than 8 distinct triples")

    used = {key for source, _, target in triples for key in (source, target)}
    nodes = [{"node_id": key, "name": attrs[key]["name"],
              "kind": attrs[key]["kind"], "topic": attrs[key]["kind"],
              "description": f"Hetionet node category: {attrs[key]['kind']}. "
                             "TSV provides no narrative description."}
             for key in sorted(used)]
    edges = [{"source": s, "target": t, "weight": 1.0, "relation": r}
             for s, r, t in sorted(triples)]
    pl.DataFrame(nodes).write_csv(output / "nodes.csv")
    pl.DataFrame(edges).write_csv(output / "edges.csv")
    result = {"focus_id": focal, "focus_name": focus, "nodes": len(nodes),
            "edges": len(edges), "output": str(output), "types": sorted({r["kind"] for r in nodes}),
            "requested_max_nodes": max_nodes, "seed": seed,
            "schema_version": 4, "triple_preserving": True,
            "note": "Biomedical KG research only; SIF has no per-edge licensing metadata."}
    (output / "dataset_meta.json").write_text(
        __import__("json").dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description="Sample local Hetionet SIF into a compact KG")
    p.add_argument("--nodes", type=Path, required=True)
    p.add_argument("--edges", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/hetionet_demo"))
    p.add_argument("--focus", default="Alzheimer's disease")
    p.add_argument("--max-nodes", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    import json
    print(json.dumps(convert_hetionet(a.nodes, a.edges, a.output, focus=a.focus,
                                     max_nodes=a.max_nodes, seed=a.seed), indent=2))


if __name__ == "__main__":
    main()
