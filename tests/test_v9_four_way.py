"""v9 four-mode search/PDF tests. No Ollama server or medical data required."""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from gnn_rag.comparison_pdf import FOUR_WAY_PDF, generate_four_way_pdf
from gnn_rag.grounded import grounded_search
from gnn_rag.llm import OllamaClient


class FakeIndex:
    node_ids = ('A', 'B', 'C', 'D', 'E')
    node_to_row = {name: i for i, name in enumerate(('A', 'B', 'C', 'D', 'E'))}


class FakeRanker:
    def known_edge_score(self, source, relation, target):
        return .9 if target == 'C' else .2


def test_bfs_gnn_fusion_are_distinct_and_only_use_observed_edges():
    known = pl.DataFrame([
        {'source': 'A', 'target': 'B', 'relation': 'r'},
        {'source': 'A', 'target': 'C', 'relation': 'r'},
        {'source': 'B', 'target': 'D', 'relation': 'r'},
        {'source': 'B', 'target': 'E', 'relation': 'r'},
    ])
    kwargs = dict(index=FakeIndex(), known_edges=known, start_id='A', hops=1,
                  top_k=1, beam_width=5, edge_ranker=FakeRanker())
    bfs = grounded_search(**kwargs, ranking_method='bfs_only')
    gnn = grounded_search(**kwargs, ranking_method='gnn_only')
    fusion_b = grounded_search(**kwargs, ranking_method='bfs_gnn', fusion_alpha=0.)
    fusion_g = grounded_search(**kwargs, ranking_method='bfs_gnn', fusion_alpha=1.)
    assert bfs[0].nodes == ('A','B')
    assert gnn[0].nodes == ('A','C')
    assert fusion_b[0].nodes == bfs[0].nodes
    assert fusion_g[0].nodes == gnn[0].nodes
    assert all(p.edges[0].source == 'A' and p.edges[0].target in ('B','C')
               for p in bfs + gnn + fusion_b + fusion_g)


def _payload():
    def entry(answer, *, unverified=False):
        return {'answer': answer, 'status': 'unverified_generation' if unverified else 'selected',
                'selected_evidence_ids': [] if unverified else ['E1'],
                'catalog': [] if unverified else [{'evidence_id':'E1',
                          'statement':'Alzheimer’s diseaseとParkinson’s diseaseは関連する。'}],
                'evidence': []}
    return {'created_utc':'2026-11-15T06:00:00+00:00', 'node_name':'Alzheimer’s disease',
            'node_id':'Disease::DOID:10652', 'question':'関係を説明してください',
            'model':'qwen3:1.7b', 'hops':2,'top_k':3,'fusion_alpha':.65,
            'results':{
                'plain_qwen':entry('これはKGを参照しない自由生成の出力です。',unverified=True),
                'bfs_only':entry('BFSに基づく根拠選択'),
                'gnn_only':entry('GNNに基づく根拠選択'),
                'bfs_gnn':entry('融合ランキングに基づく根拠選択'),
            }}


def test_qualitative_four_way_pdf(tmp_path:Path):
    path=generate_four_way_pdf(tmp_path / FOUR_WAY_PDF,_payload())
    assert path.read_bytes().startswith(b'%PDF-')
    assert path.stat().st_size > 3000
    assert not path.with_suffix('.pdf.tmp').exists()


def test_fourway_pdf_requires_every_method(tmp_path:Path):
    incomplete=_payload();del incomplete['results']['bfs_only']
    with pytest.raises(ValueError,match='four'):
        generate_four_way_pdf(tmp_path / FOUR_WAY_PDF,incomplete)


def test_plain_qwen_uses_its_own_system_prompt(monkeypatch):
    from gnn_rag import llm
    import json
    seen = {}
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):return False
    def fake_urlopen(request,timeout):
        seen.update(json.loads(request.data))
        class R(Response):
            def read(self):return b'{}'
        # json.load needs a file with read method
        class Actual(Response):
            def read(self, n=-1):return b'{"message":{"content":"plain answer"}}'
        return Actual()
    monkeypatch.setattr(llm,'urlopen',fake_urlopen)
    got=OllamaClient().plain_answer('Selected node','What is it?')
    assert got=='plain answer'
    assert 'Selected node' in seen['messages'][1]['content']
    assert '知識グラフや検索の根拠は提供されません' in seen['messages'][0]['content']
    assert 'format' not in seen
