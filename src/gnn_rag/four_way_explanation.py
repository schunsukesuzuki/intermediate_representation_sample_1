"""v11: independent sequence interpretations, then four-way Qwen synthesis.

Do not conflate observed TRAIN edges with conjectured links. Cross comparison
is post-hoc commentary, never a validated prediction/clinical metric.
"""
from __future__ import annotations
from typing import Any
from .structured_paths import build_verified_paths, format_verified_paths, build_method_audit, build_cross_factual_comparison

FOUR_METHODS = ('plain_qwen', 'bfs_only', 'gnn_only', 'bfs_gnn')


def add_explanations(comparison: dict[str, Any], client: Any,
                     on_update=None) -> dict[str, Any]:
    """Add the independent, two-part Qwen commentary to each original answer."""
    for method in FOUR_METHODS:
        if on_update is not None:
            on_update(method)
        entry = comparison['results'][method]
        # This provenance-preserving intermediate representation is immutable
        # evidence, independently constructed from the selected KG candidates.
        entry['structured_paths'] = (build_verified_paths(
            entry.get('catalog', []), entry.get('evidence', []),
            entry.get('selected_evidence_ids', [])) if method != 'plain_qwen' else [])
        entry['structured_path_text'] = (format_verified_paths(entry['structured_paths'])
            if method != 'plain_qwen' else 'Plain Qwen: KG検索なし（ノード・エッジ未提供）。')
        entry['sequence_interpretation'] = ''
        entry['explanation'] = ''
        entry['explanation_status'] = 'skipped_no_answer'
        entry.pop('explanation_error', None)
        entry.pop('explanation_raw_response', None)
        entry.pop('explanation_attempts', None)
        entry.pop('generation_warning', None)
        entry.pop('interpretation_warning', None)
        answer = entry.get('answer', '')
        if not isinstance(answer, str) or not answer.strip():
            continue
        try:
            generated = client.explain_result(
                method=method, subject=comparison['node_name'],
                question=comparison['question'], answer=answer,
                answer_status=entry.get('status', 'unknown'),
                catalog=entry.get('catalog', []),
                selected_evidence_ids=entry.get('selected_evidence_ids', []),
                evidence_paths=entry.get('evidence', []),
            )
            if not isinstance(generated, dict) or any(
                not isinstance(generated.get(k), str) or not generated[k].strip()
                for k in ('sequence_interpretation', 'explanation')
            ):
                raise RuntimeError('Qwen returned an incomplete two-part explanation')
            if generated.get('generation_warning'):
                entry['generation_warning'] = generated['generation_warning']
            generated_interpretation = generated['sequence_interpretation'].strip()
            if entry['structured_paths'] and 'KGシーケンス未提供' in generated_interpretation:
                # The text model can deny that a path was supplied. Do not let
                # this overwrite the observed structure in either output.
                entry['interpretation_warning'] = (
                    'Qwenの解釈は観測済み経路と矛盾（KGシーケンス未提供と誤認）。'
                    'この解釈は採用せず、上の確定的な中間表現を参照してください。')
                entry['sequence_interpretation'] = entry['interpretation_warning']
            else:
                entry['sequence_interpretation'] = generated_interpretation
            entry['explanation'] = generated['explanation'].strip()
            entry['explanation_status'] = 'generated_unverified'
        except (RuntimeError, ValueError, TypeError) as exc:
            entry['explanation_status'] = 'error'
            entry['explanation_error'] = str(exc)
            if getattr(exc, 'raw_response', ''):
                entry['explanation_raw_response'] = str(exc.raw_response)[:4096]
            if getattr(exc, 'attempts', None):
                entry['explanation_attempts'] = list(exc.attempts)
    comparison['explanation_note'] = (
        'Each method first receives a sequence-based Qwen interpretation, including '
        'explicitly unverified hypotheses, followed by a 3-6 sentence explanation. '
        'Plain Qwen has no KG; graph methods receive retrieved observed TRAIN paths '
        'and separately marked selected facts. Neither interpretation is validated KG evidence.'
    )
    return comparison


def add_cross_comparison(comparison: dict[str, Any], client: Any) -> dict[str, Any]:
    """Cross-compare ONLY after independent explanations; preserve them on failure."""
    comparison['method_audit'] = build_method_audit(comparison['results'])
    comparison['cross_factual_comparison'] = build_cross_factual_comparison(comparison['method_audit'])
    comparison['cross_comparison'] = ''
    comparison['cross_comparison_status'] = 'not_run'
    comparison.pop('cross_comparison_error', None)
    comparison.pop('cross_comparison_raw_response', None)
    comparison.pop('cross_comparison_attempts', None)
    if not any(comparison['results'][key].get('answer') for key in FOUR_METHODS):
        comparison['cross_comparison_status'] = 'skipped_no_answers'
        return comparison
    try:
        message = client.compare_four_explanations(
            subject=comparison['node_name'], question=comparison['question'],
            results=comparison['results'],
        )
        if not isinstance(message, str) or not message.strip():
            raise RuntimeError('Qwen returned an empty cross-method comparison')
        comparison['cross_comparison'] = message.strip()
        comparison['cross_comparison_status'] = 'generated_unverified'
    except (RuntimeError, ValueError, TypeError) as exc:
        comparison['cross_comparison_status'] = 'error'
        comparison['cross_comparison_error'] = str(exc)
        if getattr(exc, 'raw_response', ''):
            comparison['cross_comparison_raw_response'] = str(exc.raw_response)[:4096]
        if getattr(exc, 'attempts', None):
            comparison['cross_comparison_attempts'] = list(exc.attempts)
    comparison['cross_comparison_note'] = (
        'The Qwen cross-method comparison receives four original answers and individual '
        'interpretations, but only produces unverified hypotheses. All selected triples '
        'are audited by Python, not the LLM; no objective QA or clinical validation.'
    )
    return comparison
