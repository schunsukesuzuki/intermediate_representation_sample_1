"""End-to-end relation-conditioned JAX GNN + Voyager / observed-edge RAG."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .artifacts import save_artifacts
from .data import load_graph
from .fusion_evaluation import compare_three, markdown_comparison
from .grounded import (evidence_catalog, evidence_prompt, grounded_search,
                       observed_training_edges, render_verified_answer)
from .llm import OllamaClient
from .retrieval import VoyagerRetriever
from .triple_learning import triple_split, train_triples
from .synthetic import write_synthetic_dataset
from .typed_search import RelationRanker
import numpy as np


def run_rag(*, output: Path, nodes_csv: Path | None = None, edges_csv: Path | None = None,
            epochs: int = 30, seed: int = 42, start_id: str | None = None,
            question: str = "この概念と関連する観測済みエッジを説明してください。",
            hops: int = 2, top_k: int = 3, beam_width: int = 8,
            generate: bool = False, model: str = "qwen3:1.7b",
            ollama_url: str = "http://127.0.0.1:11434",
            evaluate: bool = True, random_repeats: int = 20,
            eval_max_queries: int | None = None,
            train_max_positives: int | None = None,
            feature_mode: str = 'kind_name') -> dict:
    """Train once, export observed-only evidence, then separately evaluate held-out links."""
    if bool(nodes_csv) != bool(edges_csv):
        raise ValueError("Supply both nodes_csv and edges_csv, or neither for synthetic")
    output.mkdir(parents=True, exist_ok=True)
    synthetic = nodes_csv is None
    if synthetic:
        nodes_csv, edges_csv = write_synthetic_dataset(output / "synthetic")
    graph = load_graph(nodes_csv, edges_csv)
    split = triple_split(graph, seed=seed)
    trained = train_triples(graph, split, epochs=epochs, seed=seed,
                            max_positives_per_epoch=train_max_positives,
                            feature_mode=feature_mode)
    save_artifacts(output, graph, trained)
    relation_vectors = np.asarray(trained.params['relation_vectors'], dtype=np.float32)
    np.save(output / 'relation_vectors.npy', relation_vectors, allow_pickle=False)
    (output / 'relation_names.json').write_text(
        json.dumps(graph.relation_names, ensure_ascii=False), encoding='utf-8')
    (output / 'node_kinds.json').write_text(
        json.dumps(graph.node_kinds, ensure_ascii=False), encoding='utf-8')
    ranker = RelationRanker(graph.ids, graph.node_kinds, graph.relation_names,
                            trained.embeddings, relation_vectors)
    observed = observed_training_edges(graph.edges, graph.ids, split.train[:, [0, 2]])
    observed.write_csv(output / "train_edges.csv")
    index = VoyagerRetriever(graph.ids, trained.embeddings)
    chosen = start_id
    if chosen is None and not synthetic:
        meta = nodes_csv.parent / "dataset_meta.json"
        if meta.exists():
            chosen = json.loads(meta.read_text(encoding="utf-8")).get("focus_id")
    chosen = chosen or graph.ids[0]
    paths = grounded_search(index, observed, chosen, hops=hops,
                            top_k=top_k, beam_width=beam_width,
                            edge_ranker=ranker)
    facts = evidence_catalog(graph.nodes, paths)
    prompt = evidence_prompt(graph.nodes, paths, question)
    (output / "grounded_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    payload = [{"nodes": p.nodes, "distance": p.distance,
                "edges": [vars(edge) for edge in p.edges]} for p in paths]
    (output / "evidence_paths.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "evidence_catalog.json").write_text(
        json.dumps([asdict(f) for f in facts], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    evaluation = None
    if evaluate:
        evaluation, rows = compare_three(
            graph, split, ranker, seed=seed, max_queries=eval_max_queries,
        )
        evaluation.update({"dataset": "synthetic" if synthetic else "hetionet_subset",
                           "split_seed": seed, "epochs": epochs,
                           "train_triples": len(split.train),
                           "validation_triples": len(split.validation),
                           "test_triples": len(split.test),
                           "relation_count": len(graph.relation_names)})
        (output / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "evaluation.md").write_text(markdown_comparison(evaluation), encoding="utf-8")
        with (output / "evaluation_per_query.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    answer = None
    selection_status = "not_requested"
    if generate:
        allowed_ids = tuple(f.evidence_id for f in facts)
        selected = OllamaClient(model=model, endpoint=ollama_url).select_evidence(prompt, allowed_ids)
        answer = render_verified_answer(facts, selected.ids)
        selection_status = selected.status
        (output / "llm_answer.txt").write_text(answer + "\n", encoding="utf-8")
    result = {"dataset": "synthetic" if synthetic else ("hetionet_subset" if (nodes_csv.parent / "dataset_meta.json").exists() else "user_csv"),
              "start_id": chosen, "observed_edges": observed.height,
              "metrics": trained.metrics, "triple_aware": True,
              "relations": list(graph.relation_names), "evidence_path_count": len(paths),
              "verified_evidence_count": len(facts),
              "evaluation": {"report": "evaluation.json", "query_nodes": evaluation["query_count"]}
              if evaluation is not None else None,
              "generated": generate, "selection_status": selection_status,
              "model": model if generate else None, "answer": answer, "output": str(output)}
    (output / "run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description="Train a compact KG GNN-RAG and optionally call local Ollama")
    p.add_argument("--nodes", type=Path)
    p.add_argument("--edges", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/local_rag"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-id")
    p.add_argument("--question", default="この概念と関連する観測済みエッジを説明してください。")
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--beam-width", type=int, default=8)
    p.add_argument("--generate", action="store_true", help="Actually call local Ollama")
    p.add_argument("--model", default="qwen3:1.7b")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--no-evaluate", action="store_true")
    p.add_argument("--random-repeats", type=int, default=20)
    p.add_argument('--eval-max-queries', type=int)
    p.add_argument('--train-max-positives', type=int)
    p.add_argument('--feature-mode', choices=['kind_name', 'kind_only'], default='kind_name')
    args = p.parse_args()
    result = run_rag(output=args.output, nodes_csv=args.nodes, edges_csv=args.edges,
                     epochs=args.epochs, seed=args.seed, start_id=args.start_id,
                     question=args.question, hops=args.hops, top_k=args.top_k,
                     beam_width=args.beam_width, generate=args.generate,
                     model=args.model, ollama_url=args.ollama_url,
                     evaluate=not args.no_evaluate,
                     random_repeats=args.random_repeats,
                     eval_max_queries=args.eval_max_queries,
                     train_max_positives=args.train_max_positives,
                     feature_mode=args.feature_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Review typed evidence prompt: {args.output / 'grounded_prompt.txt'}")
    print(f"Review held-out retrieval evaluation: {args.output / 'evaluation.md'}")


if __name__ == "__main__":
    main()
