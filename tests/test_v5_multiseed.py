import pytest
from gnn_rag.multiseed import summarize_reports, markdown_multiseed


def _run(seed, values):
    metrics = {m: {'test': {k: v for k in ('hits@1', 'hits@3', 'hits@10',
                                        'mrr@3', 'mrr@10', 'recall@10')}}
               for m, v in values.items()}
    return {'seed': seed, 'methods': metrics}


def test_summary_uses_paired_deltas_and_sample_std():
    runs = [_run(41, {'gnn_only': .4, 'bfs_only': .2, 'bfs_gnn': .5}),
            _run(42, {'gnn_only': .5, 'bfs_only': .3, 'bfs_gnn': .1})]
    result = summarize_reports(runs)
    assert result['methods']['gnn_only']['mrr@10']['mean'] == pytest.approx(.45)
    assert result['paired_mrr10_delta']['bfs_gnn_minus_bfs_only']['per_seed'] == pytest.approx([.3, -.2])
    assert result['paired_mrr10_delta']['bfs_gnn_minus_bfs_only']['positive_seeds'] == 1
    assert 'multi-seed' in markdown_multiseed(result)


def test_reject_duplicate_seed():
    r = _run(41, {'gnn_only': .4, 'bfs_only': .2, 'bfs_gnn': .5})
    with pytest.raises(ValueError, match='Repeated seed'):
        summarize_reports([r, r])
