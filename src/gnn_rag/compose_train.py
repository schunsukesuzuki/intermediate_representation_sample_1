"""One-shot Docker Compose trainer; production API never trains on request.

The dataset is selected explicitly to avoid accidentally falling back to the
synthetic graph when public-data inputs are missing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path



DEFAULT_DATA_ROOT = Path("/app/data/hetionet_demo")
DEFAULT_OUTPUT = Path("/app/outputs/local_rag")


def training_options(environ: dict[str, str] | None = None) -> dict:
    env = os.environ if environ is None else environ
    dataset = env.get("GNN_RAG_DATASET", "synthetic").strip().lower()
    if dataset not in ("synthetic", "hetionet"):
        raise ValueError("GNN_RAG_DATASET must be 'synthetic' or 'hetionet'")
    epochs = int(env.get("GNN_RAG_EPOCHS", "25"))
    seed = int(env.get("GNN_RAG_SEED", "42"))
    random_repeats = int(env.get("GNN_RAG_RANDOM_REPEATS", "20"))
    feature_mode = env.get('GNN_RAG_FEATURE_MODE', 'kind_name').strip()
    if feature_mode not in ('kind_only', 'kind_name'):
        raise ValueError('GNN_RAG_FEATURE_MODE must be kind_only or kind_name')
    eval_max_queries = int(env.get('GNN_RAG_EVAL_MAX_QUERIES', '0'))
    train_max_positives = int(env.get('GNN_RAG_TRAIN_MAX_POSITIVES', '0'))
    if eval_max_queries < 0 or train_max_positives < 0:
        raise ValueError('Evaluation/training sample limits must be non-negative')
    if epochs < 1 or epochs > 10000:
        raise ValueError("GNN_RAG_EPOCHS must be between 1 and 10000")
    if seed < 0:
        raise ValueError("GNN_RAG_SEED must be nonnegative")
    if not 1 <= random_repeats <= 1000:
        raise ValueError("GNN_RAG_RANDOM_REPEATS must be 1..1000")
    result = {"output": DEFAULT_OUTPUT, "epochs": epochs, "seed": seed,
              "random_repeats": random_repeats,
              'feature_mode': feature_mode,
              'eval_max_queries': eval_max_queries or None,
              'train_max_positives': train_max_positives or None}
    if dataset == "hetionet":
        nodes = DEFAULT_DATA_ROOT / "nodes.csv"
        edges = DEFAULT_DATA_ROOT / "edges.csv"
        if not nodes.is_file() or not edges.is_file():
            raise FileNotFoundError(
                "Hetionet CSVs missing after prepare-data. Run "
                "`docker compose logs --no-color prepare-data` to diagnose preparation, "
                "and verify ./data is mounted at /app/data in both services."
            )
        result.update(nodes_csv=nodes, edges_csv=edges)
    else:
        result["start_id"] = "n00"
    return result


def main() -> None:
    from .rag import run_rag  # large ML imports only when the actual job starts
    result = run_rag(**training_options())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Training complete. Generated evidence and model files are under outputs/local_rag/.")


if __name__ == "__main__":
    main()
