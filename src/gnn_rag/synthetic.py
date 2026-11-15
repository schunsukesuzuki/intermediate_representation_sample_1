"""Invented ontology, not a patient record or an extracted real-world KG."""

from pathlib import Path

import polars as pl

# Cross-links deliberately make this an interesting but small link-prediction demo.
_NODES = [
    ("n00", "看護記録", "records", "artifact", "看護師が記述した経過記録"),
    ("n01", "経過観察", "records", "process", "状態変化を継続して把握する作業"),
    ("n02", "記録の欠損", "records", "risk", "記録すべき項目が空欄になっている状態"),
    ("n03", "時系列整合性", "records", "quality", "記録上の時間関係が矛盾しないこと"),
    ("n04", "転倒リスク", "safety", "risk", "転倒につながる可能性がある状態"),
    ("n05", "安全確認", "safety", "process", "リスクに応じた確認手順"),
    ("n06", "インシデント報告", "safety", "artifact", "安全上の事象を記述する報告"),
    ("n07", "誤薬防止", "safety", "process", "与薬の取り違えを抑える確認"),
    ("n08", "薬剤情報", "medication", "artifact", "薬剤名や投与条件の情報"),
    ("n09", "アレルギー確認", "medication", "process", "既知のアレルギー情報を参照する作業"),
    ("n10", "投薬記録", "medication", "artifact", "投薬の実施状況を示す記録"),
    ("n11", "投薬時刻", "medication", "quality", "投薬タイミングの記録の整合性"),
    ("n12", "バイタルサイン", "monitoring", "measurement", "体温や脈拍などの観測項目"),
    ("n13", "異常値アラート", "monitoring", "process", "閾値逸脱を知らせる仕組み"),
    ("n14", "モニタリング", "monitoring", "process", "観測値を継続的に確認する作業"),
    ("n15", "再測定", "monitoring", "process", "観測値を確認のため測り直す作業"),
    ("n16", "記録品質評価", "records", "quality", "記録の網羅性と整合性を確認する作業"),
    ("n17", "看護申し送り", "records", "process", "看護者間の情報引き継ぎ"),
    ("n18", "服薬指導", "medication", "process", "薬剤の使用方法を説明する作業"),
    ("n19", "リスク評価", "safety", "quality", "複数の危険因子を整理する判断"),
]

# source, target; direction is preserved for link-prediction labels.
_RELATIONS = [
    (0, 1), (0, 2), (0, 3), (0, 16), (1, 12), (1, 14), (1, 17),
    (2, 16), (3, 16), (3, 11), (4, 5), (4, 19), (5, 6), (5, 7),
    (6, 17), (7, 8), (7, 9), (8, 9), (8, 10), (8, 18), (9, 10),
    (10, 11), (10, 18), (11, 3), (12, 13), (12, 14), (13, 15),
    (14, 15), (14, 19), (15, 1), (16, 17), (17, 5), (18, 7),
    (19, 6), (19, 4), (2, 17), (13, 5), (4, 14), (11, 16), (9, 19),
]


def write_synthetic_dataset(directory: Path | str) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    nodes_path, edges_path = directory / "nodes.csv", directory / "edges.csv"
    pl.DataFrame(_NODES, schema=["node_id", "name", "topic", "kind", "description"], orient="row").write_csv(nodes_path)
    pl.DataFrame(
        [(f"n{s:02d}", f"n{t:02d}", 1.0) for s, t in _RELATIONS],
        schema=["source", "target", "weight"], orient="row",
    ).write_csv(edges_path)
    return nodes_path, edges_path
