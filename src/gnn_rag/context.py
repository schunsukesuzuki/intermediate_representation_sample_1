"""Build inspectable retrieval context, with no external LLM or vector DB."""

from __future__ import annotations

from typing import Sequence

import polars as pl

from .retrieval import SearchPath


def make_context(nodes: pl.DataFrame, paths: Sequence[SearchPath], *, question: str,
                 max_paths: int = 5) -> str:
    """Return a draft RAG prompt. ANN paths are NOT verified KG-edge citations."""
    if max_paths < 1:
        raise ValueError("max_paths must be >= 1")
    lookup = {row["node_id"]: row for row in nodes.iter_rows(named=True)}
    lines = [
        "あなたは架空の看護業務オントロジーを説明するアシスタントです。",
        "以下は説明用の候補情報であり、医学的判断や事実の根拠ではありません。",
        "ANNのホップは埋め込み空間上の類似検索です。KGの実エッジを保証しません。",
        "根拠が不足する場合は不明と答え、候補情報を断定しないでください。",
        "", "【検索コンテキスト】",
    ]
    for i, path in enumerate(paths[:max_paths], start=1):
        titles = " → ".join(lookup[node_id]["name"] for node_id in path.node_ids)
        lines.append(f"候補{i}: {titles} (ANN距離合計={path.score:.4f})")
        for node_id in path.node_ids:
            row = lookup[node_id]
            lines.append(f"  - {row['name']}: {row['description']}")
    lines.extend(["", f"質問: {question}", "回答（LLM未接続のため生成しません）:"])
    return "\n".join(lines)
