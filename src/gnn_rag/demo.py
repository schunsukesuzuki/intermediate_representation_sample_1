"""CLI: generate imaginary KG, train JAX GNN, index Voyager, build context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import save_artifacts
from .context import make_context
from .data import load_graph
from .retrieval import VoyagerRetriever
from .split import split_edges
from .synthetic import write_synthetic_dataset
from .train import train


def run_demo(*, output: Path, epochs: int, seed: int,
             start_id: str, hops: int, top_k: int, beam_width: int) -> dict:
    nodes_path, edges_path = write_synthetic_dataset(output / "synthetic")
    graph = load_graph(nodes_path, edges_path)
    split = split_edges(graph, seed=seed)
    result = train(graph, split, seed=seed, epochs=epochs)
    save_artifacts(output, graph, result)
    retriever = VoyagerRetriever(graph.ids, result.embeddings)
    paths = retriever.multi_hop(start_id, hops=hops, top_k=top_k, beam_width=beam_width)
    question = "看護記録の品質改善に関連する概念を整理してください。"
    context = make_context(graph.nodes, paths, question=question)
    (output / "prompt_preview.txt").write_text(context + "\n", encoding="utf-8")
    trace = [{"node_ids": p.node_ids, "hop_distances": p.distances,
              "total_ann_distance": p.score} for p in paths]
    (output / "retrieval_paths.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return {"metrics": result.metrics, "paths": trace, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-id", default="n00")
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=8)
    args = parser.parse_args()
    result = run_demo(output=args.output, epochs=args.epochs, seed=args.seed,
                      start_id=args.start_id, hops=args.hops, top_k=args.top_k,
                      beam_width=args.beam_width)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nPrompt preview:", args.output / "prompt_preview.txt")
    print("NOTE: metrics are from synthetic data and are not performance claims.")


if __name__ == "__main__":
    main()
