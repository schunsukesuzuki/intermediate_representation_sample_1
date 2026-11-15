"""Generate a reproducible, self-contained PDF from newly computed benchmark metrics.

The PDF is generated from `overview.json` only; it never substitutes bundled
reference scores for the metrics produced by the current benchmark run.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, KeepTogether, PageBreak)
from reportlab.graphics.shapes import Drawing, Rect, String, Line

REPORT_NAME = 'Hetionet_10k_100k_comparison.pdf'
_METHODS = [('gnn_only', 'GNN only'), ('bfs_only', 'BFS only'), ('bfs_gnn', 'BFS + GNN')]
_NAVY = colors.HexColor('#12243E')
_BLUE = colors.HexColor('#2367A5')
_TEAL = colors.HexColor('#0B847B')
_GOLD = colors.HexColor('#B77A29')
_MUTED = colors.HexColor('#64748B')
_BORDER = colors.HexColor('#DCE5ED')
_BG = colors.HexColor('#F3F6FA')


def _font() -> str:
    """Embed IPA Gothic in Docker; use built-in Japanese CID as a fallback."""
    for font_path in (
        '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
        '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf',
        '/usr/share/fonts/truetype/ipafont-gothic/ipag.ttf',
    ):
        if Path(font_path).is_file():
            if 'IPA-v8' not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont('IPA-v8', font_path))
            return 'IPA-v8'
    if 'HeiseiKakuGo-W5' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
    return 'HeiseiKakuGo-W5'


def _fmt(value: dict | None) -> str:
    if not value:
        return '—'
    mean = float(value['mean'])
    sd = value.get('sd')
    return f'{mean:.4f} ± {float(sd):.4f}' if sd is not None else f'{mean:.4f} (1 seed)'


def _paragraph(txt: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(txt)).replace('\n', '<br/>'), style)


def _chart(sizes: dict, font: str) -> Drawing:
    """Side-by-side exact-ranking MRR@10 bars, independently scaled groups."""
    drawing = Drawing(475, 150)
    maxv = max((float(s['methods'][m]['mrr@10']['mean'])
                for s in sizes.values() for m, _ in _METHODS), default=0.1)
    maxv = max(.05, maxv * 1.15)
    cols = [_BLUE, _MUTED, _TEAL]
    blocks = sorted(sizes.items(), key=lambda x: int(x[0]))
    nblocks = max(1, len(blocks))
    w = 475 / nblocks
    for j, (size, data) in enumerate(blocks):
        left = j*w + 16
        drawing.add(String(left, 139, f'{int(size):,} triples', fontName=font,
                           fontSize=11, fillColor=_NAVY))
        drawing.add(Line(left, 27, left+w-40, 27, strokeColor=_BORDER))
        bw = min(29, (w-55)/4)
        for i, (method, name) in enumerate(_METHODS):
            val = float(data['methods'][method]['mrr@10']['mean'])
            h = min(93, (val/maxv)*93)
            x = left+10+i*(bw+20)
            drawing.add(Rect(x, 28, bw, h, rx=3, ry=3, fillColor=cols[i], strokeColor=None))
            drawing.add(String(x+bw/2, 31+h, f'{val:.3f}', fontSize=8, textAnchor='middle',
                               fillColor=_NAVY))
            drawing.add(String(x+bw/2, 14, ['GNN', 'BFS', 'Fusion'][i],
                               fontSize=8, textAnchor='middle', fillColor=_MUTED))
    return drawing


def generate_pdf_report(output: Path, report: dict) -> Path:
    """Build full report, atomically replacing output only when complete.

    `report` is the fresh `overview.json`: {parameters: ..., sizes: ...}.
    """
    if not isinstance(report.get('sizes'), dict) or not report['sizes']:
        raise ValueError('A computed benchmark summary is required for the PDF')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    font = _font()
    title = ParagraphStyle('TitleV8', fontName=font, fontSize=20, leading=28,
                           textColor=_NAVY, spaceAfter=8)
    h2 = ParagraphStyle('HeadingV8', fontName=font, fontSize=13, leading=19,
                        textColor=_NAVY, spaceBefore=10, spaceAfter=7)
    normal = ParagraphStyle('NormalV8', fontName=font, fontSize=9.2, leading=15,
                            textColor=_NAVY, spaceAfter=6, wordWrap='CJK')
    caption = ParagraphStyle('CaptionV8', fontName=font, fontSize=8, leading=12,
                             textColor=_MUTED, spaceAfter=7, wordWrap='CJK')
    th = ParagraphStyle('TableHeaderV8', fontName=font, fontSize=8.1, leading=12,
                        textColor=colors.white, wordWrap='CJK')
    cell = ParagraphStyle('TableCellV8', fontName=font, fontSize=8, leading=12,
                          textColor=_NAVY, wordWrap='CJK')

    params = report.get('parameters') or {}
    sizes = report['sizes']
    tmp = output.with_suffix('.pdf.tmp')
    doc = BaseDocTemplate(str(tmp), pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
                          topMargin=19*mm, bottomMargin=19*mm,
                          title='Hetionet 10k/100k Training and Retrieval Comparison',
                          author='GNN-RAG v8 automatic benchmark')
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, showBoundary=0)

    def chrome(canvas, document):
        canvas.saveState()
        canvas.setFillColor(_NAVY)
        canvas.rect(0, A4[1]-10*mm, A4[0], 10*mm, stroke=0, fill=1)
        canvas.setFont(font, 8)
        canvas.setFillColor(_MUTED)
        canvas.drawString(20*mm, 11*mm, 'GNN-RAG v8  /  Hetionet Research Benchmark')
        canvas.drawRightString(A4[0]-20*mm, 11*mm, f'{document.page}')
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id='normal', frames=[frame], onPage=chrome)])
    items = [Spacer(1, 11*mm), _paragraph('Hetionet 10k / 100k 学習・検索比較', title),
             _paragraph('JAX GraphSAGE · BFS · Validation-calibrated Fusion', caption)]
    created = str(params.get('created_utc', '—'))
    items.append(_paragraph('実行日時 (UTC): ' + created + '   |   種別: 今回実行した学習結果', normal))
    items.append(_paragraph(
        f"Sampling seed {params.get('sample_seed', '—')}  ·  学習seed {', '.join(map(str, params.get('seeds', [])))}  ·  "
        f"{params.get('epochs','—')} epoch  ·  正例上限 {params.get('max_positives_per_epoch','—')}/epoch  ·  "
        f"特徴量 {params.get('feature_mode','—')}", normal))

    items.append(_paragraph('テストセットのランキング結果', h2))
    headers = ['Triple数', 'ノード数', '手法', 'MRR@10 (平均 ± SD)', 'Hits@10 (平均 ± SD)']
    rows = [[_paragraph(x, th) for x in headers]]
    for size, summary in sorted(sizes.items(), key=lambda x: int(x[0])):
        for method, label in _METHODS:
            d = summary['methods'][method]
            rows.append([_paragraph(f'{int(size):,}', cell),
                         _paragraph(f"{float(summary['mean_node_count']):,.0f}", cell),
                         _paragraph(label, cell),
                         _paragraph(_fmt(d.get('mrr@10')), cell),
                         _paragraph(_fmt(d.get('hits@10')), cell)])
    table = Table(rows, colWidths=[63, 57, 78, 126, 130], hAlign='LEFT', repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0),_NAVY),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,_BG]),
        ('BOTTOMPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),8),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LINEBELOW',(0,-1),(-1,-1),.6,_BORDER),
        ('LEFTPADDING',(0,0),(-1,-1),8),
    ]))
    items += [table, Spacer(1,9*mm), _paragraph('MRR@10: 保留リンク検索の最初の正解順位', h2), _chart(sizes,font),
              _paragraph('同一Triple規模内で3方式を比較。規模間で候補数とTestクエリ集合が異なるため、'
                         '棒の高さの規模間差を学習量の因果効果と解釈しないでください。', caption)]

    items += [PageBreak(), Spacer(1, 2*mm), _paragraph('学習条件・計算実績', h2)]
    learning_rows = [[_paragraph(x,th) for x in ['Triple数','Train Triple平均','最終Train loss平均','学習＋評価時間平均 (秒)']]]
    for size, s in sorted(sizes.items(),key=lambda x:int(x[0])):
        train = s.get('training', {})
        learning_rows.append([
            _paragraph(f'{int(size):,}',cell),
            _paragraph(f"{float(train['train_triples_mean']):,.0f}" if 'train_triples_mean' in train else '—',cell),
            _paragraph(f"{float(train['final_train_loss_mean']):.4f}" if 'final_train_loss_mean' in train else '—',cell),
            _paragraph(f"{float(train['runtime_seconds_mean']):.1f}" if 'runtime_seconds_mean' in train else '—',cell),
        ])
    learn_table = Table(learning_rows, colWidths=[80,120,129,125], repeatRows=1, hAlign='LEFT')
    learn_table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),_NAVY),
                                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,_BG]),
                                    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                                    ('TOPPADDING',(0,0),(-1,-1),9),
                                    ('BOTTOMPADDING',(0,0),(-1,-1),9)]))
    items += [learn_table, _paragraph('Train lossは最終epochのsampled-softmax lossです。規模や負例分布が異なるため絶対値の直接比較には注意が必要です。',caption)]
    items.append(_paragraph('seed別 MRR@10 と融合重み', h2))
    for size, s in sorted(sizes.items(),key=lambda x:int(x[0])):
        per_seed = s['methods']['gnn_only']['mrr@10']['per_seed']
        seed_values = s['seeds']
        r = [[_paragraph(t,th) for t in ['seed','GNN','BFS','Fusion','Fusion α (GNN)']]]
        for i, seed in enumerate(seed_values):
            vals = [float(s['methods'][m]['mrr@10']['per_seed'][i]) for m,_ in _METHODS]
            alpha = s.get('alphas',[])
            r.append([_paragraph(str(seed),cell), *[_paragraph(f'{v:.4f}',cell) for v in vals],
                      _paragraph(f'{float(alpha[i]):.2f}' if i < len(alpha) else '—',cell)])
        t = Table(r, colWidths=[65,85,85,85,134], hAlign='LEFT', repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),_NAVY),
                               ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,_BG]),
                               ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                               ('TOPPADDING',(0,0),(-1,-1),4),
                               ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        items += [KeepTogether([_paragraph(f'{int(size):,} Triple / {len(per_seed)} seed',normal),t]),Spacer(1,4*mm)]

    items.append(_paragraph('解釈上の重要な制約', h2))
    for warning in (
        '1. 10kと100kはFull Hetionetから個別に抽出された部分グラフです。Test集合、候補数、関係分布が一致しません。固定Testでのlearning curveではありません。',
        '2. 同一規模内のGNN、TrainグラフBFS、BFS＋GNNは同じ候補・正解を評価します。融合係数はValidationのみで選択します。',
        '3. ベンチマークのGNN検索は厳密全候補ランキングで、Voyager/HNSWを使用しません。APIでの近似検索性能とは異なります。',
        '4. Train adjacencyとBFSは保留Tripleを使わず、無向ノードペア単位でsplitしています。一方、部分グラフの抽出はsplitより前に行われます。',
        '5. この結果はKGの保留リンク順位を評価したもので、LLMの回答品質や医学的因果、臨床的正確性を証明しません。',
    ):
        items.append(_paragraph(warning, normal))
    doc.build(items)
    tmp.replace(output)
    return output
