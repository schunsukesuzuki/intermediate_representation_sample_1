"""Human-readable 4-way question comparison PDF, generated from actual live responses.

Never place Qwen-only output in the category of KG-grounded facts. No score is
assigned to its free-form prose by the held-out link-ranking experiment.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

from .pdf_report import _font

FOUR_WAY_PDF = 'GNN_RAG_four_way_answers.pdf'
FOUR_WAY_JSON = 'GNN_RAG_four_way_answers.json'
_NAMES = (('plain_qwen', 'Qwen only (no KG)'), ('bfs_only', 'BFS only + grounded rendering'),
          ('gnn_only', 'GNN only + grounded rendering'),
          ('bfs_gnn', 'BFS + GNN + grounded rendering'))


def generate_four_way_pdf(path: Path, result: dict) -> Path:
    if not all(key in result.get('results', {}) for key, _ in _NAMES):
        raise ValueError('All four method entries are required')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    font = _font()
    normal = ParagraphStyle('FourWayText', fontName=font, fontSize=9.0, leading=14,
                            textColor=colors.HexColor('#1D3146'), wordWrap='CJK', spaceAfter=6)
    h1 = ParagraphStyle('FourWayTitle', parent=normal, fontSize=18, leading=25, spaceAfter=13)
    h2 = ParagraphStyle('FourWaySection', parent=normal, fontSize=12, leading=18,
                        textColor=colors.HexColor('#126F82'), spaceBefore=14, spaceAfter=6, keepWithNext=True)
    small = ParagraphStyle('FourWaySmall', parent=normal, fontSize=7.8, leading=12,
                           textColor=colors.HexColor('#576B7E'))
    h3 = ParagraphStyle('FourWayExplanationTitle', parent=normal, fontSize=10,
                        leading=15, textColor=colors.HexColor('#126F82'),
                        spaceBefore=9, spaceAfter=5, keepWithNext=True)

    def p(value: object, style=normal):
        return Paragraph(escape(str(value)).replace('\n', '<br/>'), style)
    story = [p('GNN-RAG v14 | Typed Paths, Verified Facts & Qwen Hypotheses', h1),
             p(f"Generated: {result.get('created_utc', datetime.now(timezone.utc).isoformat())}", small),
             p(f"Dataset: {result.get('dataset', 'not specified')}", small),
             p(f"Selected node: {result.get('node_name', '')} ({result.get('node_id', '')})"),
             p(f"Question: {result.get('question', '')}"),
             p(f"Model: {result.get('model', 'unknown')}; Hops: {result.get('hops')}; "
               f"Top K: {result.get('top_k')}; calibrated GNN weight: "
               f"{result.get('fusion_alpha')} ({'validation' if result.get('fusion_calibrated') else 'uncalibrated'})", small),
             Spacer(1, 5 * mm)]
    for key, title in _NAMES:
        entry = result['results'][key]
        facts = entry.get('catalog', [])
        story.append(p(title, h2))
        story.append(p('Status: ' + str(entry.get('status', 'unknown')) +
                       (' | No KG retrieval' if key == 'plain_qwen' else
                        f" | {len(facts)} available evidence facts"), small))
        story.append(p(entry.get('answer', '(no answer)')))
        if key != 'plain_qwen':
            story.append(p('Verified KG evidence (train edges only):', small))
            if facts:
                for f in facts:
                    story.append(p(f"{f['evidence_id']}: {f['statement']}", small))
            else:
                story.append(p('No validated evidence returned.', small))
        else:
            story.append(p('Unverified generative baseline. Not a source-backed answer.', small))
        # Authoritative structured paths precede the untrusted LLM interpretation.
        story.append(p('確定的な中間表現 / Verified typed TRAIN paths', h3))
        story.append(p(entry.get('structured_path_text',
                             'Plain Qwen: KG検索なし' if key == 'plain_qwen' else 'No verified path'), small))
        if entry.get('generation_warning'):
            story.append(p('LLM response: ' + entry['generation_warning'], small))
        if entry.get('interpretation_warning'):
            story.append(p('Integrity warning: ' + entry['interpretation_warning'], small))
        # Qwen-generated hypothetical relationships are NOT registered KG links.
        # Keep the method-specific sequence interpretation BEFORE its 3-6
        # sentence explanation, and both AFTER the original/verifiable facts.
        state = entry.get('explanation_status', 'not_run')
        if state == 'generated_unverified':
            story.append(p('ノード・エッジのシーケンス解釈と関係の仮説 / Unverified interpretation', h3))
            story.append(p(entry.get('sequence_interpretation', '（生成内容なし）')))
            story.append(p('Qwenによる生成文の説明（3〜6文） / Commentary', h3))
            story.append(p(entry.get('explanation', '（生成内容なし）')))
        else:
            story.append(KeepTogether([
                p('Qwenによる解釈・説明 / Post-hoc commentary', h3),
                p('Status: ' + state + ' (not verified KG evidence)', small),
                p('エラー: ' + str(entry.get('explanation_error', '生成されていません')), small),
                p('診断: モデル生応答（最大4096字）は同時出力するJSONに保存。', small),
            ]))
        story.append(p('注: 推測した関係は未検証です。登録エッジの存在だけでは医学的因果を証明しません。', small))
        story.append(Spacer(1, 2 * mm))
    story.append(p('4方式の根拠照合 / Deterministic provenance audit', h2))
    for key, name in _NAMES:
        method_audit = result.get('method_audit', {}).get(key, {})
        story.append(p(name + ' | ' + str(method_audit.get('source', 'not available')), small))
        if key == 'plain_qwen':
            story.append(p('KG根拠なし。Plain Qwenの主張はHetionetのTrainエッジとはみなしません。', small))
        elif not method_audit.get('selected_train_edges'):
            story.append(p('この方式では選択された登録Trainエッジはありません。', small))
        for f in method_audit.get('selected_train_edges', []):
            story.append(p(f"[{f['evidence_id']}] {f['source_name']} ({f['source_kind']}) "
                           f"--[{f['relation']}: {f['label']}]--> "
                           f"{f['target_name']} ({f['target_kind']}); 出典: {f['provenance']}", small))
    story.append(p('4方式の確定的な事実比較 / Deterministic factual differences', h2))
    facts = result.get('cross_factual_comparison', {})
    story.append(p(facts.get('summary_text', '事実比較なし'), small))
    story.append(p('4方式の横断比較 / Cross-method comparison', h2))
    story.append(p('事実の照合は直前のPython集計が確定し、Qwenは仮説と限界のみ生成します。'
                   '未検証の生成文であり、正確性の順位や臨床的妥当性を証明しません。', small))
    cross_status = result.get('cross_comparison_status', 'not_run')
    story.append(p('Cross-comparison status: ' + cross_status, small))
    if cross_status == 'generated_unverified' and result.get('cross_comparison'):
        story.append(p(result['cross_comparison']))
    else:
        story.append(p('横断比較は生成されませんでした: ' +
                       str(result.get('cross_comparison_error', cross_status)), small))
        story.append(p('診断: モデル生応答（最大4096字）は同時出力するJSONに保存。', small))
    story += [p('Evaluation protocol and limitations', h2),
              p('All four outputs use the same question and selected node. Qwen-only receives '
                'the question and node name but NO KG retrieval. The three graph methods '
                'use observed TRAIN edges and typed evidence IDs. The LLM selects IDs only; '
                'the final graph-grounded sentences are rendered deterministically.'),
              p('BFS ranks adjacent training edges by graph structure. GNN ranks observed '
                'edges with learned relation-conditioned scores. Fusion combines local '
                'rank percentiles using an alpha calibrated on validation data. This '
                'live evidence-path comparison is NOT identical to the strict held-out '
                'link-ranking benchmark.'),
              p('Each method has a node/edge sequence interpretation followed by a 3-6 sentence '
                'Qwen explanation. Retrieved observed TRAIN paths are separate from selected '
                'facts. Hypothesized links are NOT validated KG facts. Plain Qwen is given no KG.'),
              p('Finally, Qwen compares all four original answers, sequence interpretations and '
                'explanations. This is qualitative, unverified text, NOT an objective ranking.'),
              p('No automated clinical truthfulness or QA score is computed. The Qwen-only ' 
                'output does NOT have a comparable MRR@10 / Hits@10 value. These answers '
                'are for KG research demonstrations, not medical decision making.', small)]
    tmp = path.with_suffix('.pdf.tmp')
    doc = SimpleDocTemplate(str(tmp), pagesize=A4, rightMargin=18*mm,
                            leftMargin=18*mm, topMargin=15*mm, bottomMargin=15*mm,
                            title='GNN-RAG v14 Four-way Cross Comparison')
    doc.build(story)
    tmp.replace(path)
    return path
