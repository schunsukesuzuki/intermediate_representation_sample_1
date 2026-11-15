"""Auditable Hetionet v1.0 relation dictionary and conservative descriptions.

Metagraph abbreviations follow https://het.io/about/ and
https://github.com/hetio/hetionet/blob/main/describe/definitions.json .
Do not mistake co-mention, membership, or database assertions for clinical causality.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationSpec:
    source_kind: str
    verb: str
    target_kind: str
    ja_label: str
    provenance: str = "Hetionet v1.0に収録された関係"
    directed: bool = False


# Source -> relation -> target follows the Hetionet metagraph convention.
# Original text and medical conclusions are NOT generated from this dictionary.
RELATIONS: dict[str, RelationSpec] = {
    "AdG": RelationSpec("Anatomy", "downregulates", "Gene", "解剖学的部位での遺伝子発現低下"),
    "AeG": RelationSpec("Anatomy", "expresses", "Gene", "解剖学的部位での遺伝子発現"),
    "AuG": RelationSpec("Anatomy", "upregulates", "Gene", "解剖学的部位での遺伝子発現上昇"),
    "CbG": RelationSpec("Compound", "binds", "Gene", "化合物と遺伝子産物の結合"),
    "CcSE": RelationSpec("Compound", "causes", "Side Effect", "化合物の副作用としての登録"),
    "CdG": RelationSpec("Compound", "downregulates", "Gene", "化合物曝露下での発現低下"),
    "CpD": RelationSpec("Compound", "palliates", "Disease", "疾患の症状緩和への適応"),
    "CrC": RelationSpec("Compound", "resembles", "Compound", "化合物間の構造類似"),
    "CtD": RelationSpec("Compound", "treats", "Disease", "疾患修飾的治療への適応"),
    "CuG": RelationSpec("Compound", "upregulates", "Gene", "化合物曝露下での発現上昇"),
    "DaG": RelationSpec("Disease", "associates", "Gene", "疾患と遺伝子の関連"),
    "DdG": RelationSpec("Disease", "downregulates", "Gene", "疾患サンプルでの発現低下"),
    "DlA": RelationSpec("Disease", "localizes", "Anatomy", "疾患と解剖学的部位の関連（文献抄録の共起から推定）", "文献抄録での共起に基づく統計的関連。解剖学的病変の確定ではない"),
    "DpS": RelationSpec("Disease", "presents", "Symptom", "疾患と症状の関連（文献抄録の共起から推定）", "文献抄録での共起に基づく統計的関連。必発症状の主張ではない"),
    "DrD": RelationSpec("Disease", "resembles", "Disease", "疾患間の類似（文献抄録の共起から推定）", "文献抄録での共起に基づく統計的類似。病態・因果関係の実証ではない"),
    "DuG": RelationSpec("Disease", "upregulates", "Gene", "疾患サンプルでの発現上昇"),
    "GcG": RelationSpec("Gene", "covaries", "Gene", "遺伝子間の進化速度の共変動"),
    "GiG": RelationSpec("Gene", "interacts", "Gene", "遺伝子産物間の相互作用"),
    "GpBP": RelationSpec("Gene", "participates", "Biological Process", "遺伝子と生物学的プロセスの関係"),
    "GpCC": RelationSpec("Gene", "participates", "Cellular Component", "遺伝子と細胞内構成要素の関係"),
    "GpMF": RelationSpec("Gene", "participates", "Molecular Function", "遺伝子と分子機能の関係"),
    "GpPW": RelationSpec("Gene", "participates", "Pathway", "遺伝子とパスウェイの関係"),
    "Gr>G": RelationSpec("Gene", "regulates", "Gene", "遺伝子間の方向付き調節関係", directed=True),
    "PCiC": RelationSpec("Pharmacologic Class", "includes", "Compound", "薬理学的分類への化合物の所属"),
}


def relation_details(code: str, source_kind: str, target_kind: str) -> dict:
    """Validate types and return schema facts (not LLM guesses).

    Synthetic 'related_to' is intentionally untyped, not a medical assertion.
    Unknown Hetionet codes are rejected for grounded explanations.
    """
    if code == "related_to":
        return {"code": code, "label": "型未指定の関係（合成データ）",
                "source_kind": source_kind, "target_kind": target_kind,
                "provenance": "合成データの登録エッジ。医学的主張ではない", "valid": True,
                "directed": False}
    spec = RELATIONS.get(code)
    if spec is None:
        return {"code": code, "label": "未定義の関係コード", "source_kind": source_kind,
                "target_kind": target_kind, "provenance": "未確認", "valid": False,
                "directed": False}
    valid = source_kind == spec.source_kind and target_kind == spec.target_kind
    return {"code": code, "label": spec.ja_label,
            "source_kind": source_kind, "target_kind": target_kind,
            "provenance": spec.provenance, "valid": valid, "directed": spec.directed}
