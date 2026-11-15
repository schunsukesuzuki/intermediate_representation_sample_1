"""Optional read-only FastAPI adapter around precomputed artifacts.

Run the training CLI once before starting this server. API inference never
trains; /answer calls local Ollama (loopback or internal Compose network). Never expose to patients.
Inputs are public/synthetic node IDs, not patient text.
"""

from __future__ import annotations

import os
import json
from urllib.request import urlopen
from functools import lru_cache
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .artifacts import load_artifacts
from .context import make_context
from .grounded import evidence_catalog, evidence_prompt, grounded_search, render_verified_answer
from .comparison_pdf import generate_four_way_pdf, FOUR_WAY_PDF, FOUR_WAY_JSON
from .four_way_explanation import add_explanations, add_cross_comparison
from .llm import OllamaClient
from .comparison_jobs import ComparisonJobs
from .retrieval import VoyagerRetriever
from .typed_search import RelationRanker
from .fusion_evaluation import rank_candidates
from .ontology import RELATIONS
import numpy as np


class RetrievalRequest(BaseModel):
    node_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=500)
    hops: int = Field(default=2, ge=1, le=8)
    top_k: int = Field(default=3, ge=1, le=10)
    beam_width: int = Field(default=8, ge=1, le=32)
    relation: str | None = None  # optional exact Hetionet relation code; no NLP guessing


class TripleQuery(BaseModel):
    node_id: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    method: str = Field(default="bfs_gnn", pattern="^(gnn_only|bfs_only|bfs_gnn)$")


def create_app(artifacts_dir: Path | str | None = None) -> FastAPI:
    directory = Path(artifacts_dir or os.environ.get("GNN_RAG_ARTIFACTS", "outputs/demo"))
    api = FastAPI(title="GNN-RAG triple-aware demo", version="0.14.0")
    jobs = ComparisonJobs()

    @lru_cache(maxsize=1)
    def resources() -> tuple[VoyagerRetriever, pl.DataFrame]:
        node_ids, embedded = load_artifacts(directory)
        nodes = pl.read_csv(directory / "nodes.csv")
        if tuple(nodes.get_column("node_id").to_list()) != node_ids:
            raise ValueError("Artifact node order mismatch")
        return VoyagerRetriever(node_ids, embedded), nodes

    @lru_cache(maxsize=1)
    def relation_ranker() -> RelationRanker:
        node_ids, embeddings = load_artifacts(directory)
        vectors = np.load(directory / 'relation_vectors.npy', allow_pickle=False)
        relation_names = tuple(json.loads((directory / 'relation_names.json').read_text(encoding='utf-8')))
        node_kinds = tuple(json.loads((directory / 'node_kinds.json').read_text(encoding='utf-8')))
        return RelationRanker(node_ids, node_kinds, relation_names, embeddings, vectors)

    @api.get("/health")
    def health() -> dict:
        available = all((directory / name).exists()
                        for name in ("embeddings.npy", "nodes.csv", "node_ids.json",
                                     "relation_vectors.npy", "relation_names.json",
                                     "node_kinds.json", "run_summary.json"))
        expected = os.environ.get('GNN_RAG_EXPECT_DATASET', '').strip().lower()
        actual = None
        summary_path = directory / 'run_summary.json'
        if summary_path.is_file():
            try:
                actual = json.loads(summary_path.read_text(encoding='utf-8')).get('dataset')
            except (OSError, ValueError):
                pass
        dataset_matches = not expected or ((expected == 'hetionet' and actual == 'hetionet_subset') or
                                            (expected == 'synthetic' and actual == 'synthetic'))
        return {"ok": available and dataset_matches,
                "artifact_directory": str(directory), "dataset": actual,
                "expected_dataset": expected or None, "dataset_matches": dataset_matches}

    @api.get("/info")
    def info() -> dict:
        """Summary and start node for the current synthetic or Hetionet run."""
        path = directory / "run_summary.json"
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Prepare demo artifacts first") from exc
        return {"dataset": summary["dataset"], "start_id": summary["start_id"],
                "observed_edges": summary["observed_edges"],
                "triple_aware": summary.get("triple_aware", False),
                "relations": summary.get("relations", []),
                "model": os.environ.get("GNN_RAG_OLLAMA_MODEL", "qwen3:1.7b")}

    @api.get('/ollama-status')
    def ollama_status() -> dict:
        """Non-generative diagnostic; does not imply any model uses Metal GPU."""
        from urllib.parse import urlsplit
        endpoint = os.environ.get('GNN_RAG_OLLAMA_URL', 'http://host.docker.internal:11434').rstrip('/')
        model = os.environ.get('GNN_RAG_OLLAMA_MODEL', 'qwen3:1.7b')
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'http' or parsed.hostname not in (
                'localhost', '127.0.0.1', '::1', 'ollama', 'host.docker.internal') or parsed.port != 11434:
            return {'reachable': False, 'model_present': False, 'model': model,
                    'error': 'Invalid local Ollama URL; use host.docker.internal:11434 for Metal mode'}
        try:
            with urlopen(endpoint + '/api/tags', timeout=4) as response:
                payload = json.load(response)
            names = [m.get('name', '') for m in payload.get('models', []) if isinstance(m, dict)]
            present = model in names
            return {'reachable': True, 'model_present': present, 'model': model,
                    'endpoint': endpoint,
                    'note': ('Confirm actual Metal offload on the macOS host with `ollama ps`. '
                             'This HTTP endpoint cannot verify GPU usage.')}
        except Exception as exc:
            return {'reachable': False, 'model_present': False, 'model': model,
                    'endpoint': endpoint, 'error': str(exc)[:300]}

    @api.get("/evaluation")
    def evaluation() -> dict:
        """Held-out test link metrics, computed once by the trainer (not clinical QA)."""
        path = directory / "evaluation.json"
        if not path.is_file():
            raise HTTPException(status_code=503, detail="No evaluation artifact; retrain v5")
        return json.loads(path.read_text(encoding="utf-8"))

    @api.get("/benchmark-summary")
    def benchmark_summary() -> dict:
        """Read-only comparison report produced by the optional study job."""
        path = Path(os.environ.get("GNN_RAG_BENCHMARKS_DIR", "/app/outputs/benchmark")) / "overview.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Run docker compose --profile study run --rm study first")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Benchmark summary is unreadable") from exc

    @api.get("/benchmark-report")
    def benchmark_report():
        """Download the newly generated two-scale PDF, not bundled reference results."""
        from .pdf_report import REPORT_NAME
        path = Path(os.environ.get("GNN_RAG_BENCHMARKS_DIR", "/app/outputs/benchmark")) / REPORT_NAME
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Run the study job to produce a PDF report")
        return FileResponse(path, media_type="application/pdf", filename=REPORT_NAME)

    @api.get("/relations")
    def relations() -> list[dict]:
        model = relation_ranker()
        return [dict(code=code, source_kind=RELATIONS[code].source_kind,
                     target_kind=RELATIONS[code].target_kind,
                     label=RELATIONS[code].ja_label) if code in RELATIONS else
                dict(code=code, source_kind=None, target_kind=None, label='Unspecified')
                for code in model.relations]

    @api.get("/nodes")
    def nodes() -> list[dict]:
        try:
            _, table = resources()
            return table.select("node_id", "name", "topic").to_dicts()
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Prepare demo artifacts first") from exc

    @api.post("/evidence")
    def evidence(request: RetrievalRequest) -> dict:
        try:
            index, table = resources()
            observed_path = directory / "train_edges.csv"
            if not observed_path.exists():
                raise FileNotFoundError("Run python -m gnn_rag.rag first")
            known = pl.read_csv(observed_path)
            if request.relation is not None and request.relation not in relation_ranker().rel_to_row:
                raise ValueError(f'Unknown relation code: {request.relation}')
            paths = grounded_search(index, known, request.node_id,
                                    hops=request.hops, top_k=request.top_k,
                                    beam_width=request.beam_width,
                                    edge_ranker=relation_ranker(),
                                    relation_filter=request.relation)
            prompt = evidence_prompt(table, paths, request.question)
            catalog = evidence_catalog(table, paths)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"evidence": [{"nodes": p.nodes, "distance": p.distance,
                              "edges": [vars(e) for e in p.edges]} for p in paths],
                "catalog": [asdict(f) for f in catalog],
                "prompt": prompt,
                "note": "Only training edges, validated against type/relation schema, can be used as answer evidence."}

    @api.post("/answer")
    def answer(request: RetrievalRequest) -> dict:
        """Only JSON evidence IDs come from the LLM; final sentences are deterministic."""
        retrieval = evidence(request)
        catalog = retrieval["catalog"]
        from .grounded import EvidenceFact
        facts = [EvidenceFact(**record) for record in catalog]
        try:
            model = os.environ.get("GNN_RAG_OLLAMA_MODEL", "qwen3:1.7b")
            endpoint = os.environ.get("GNN_RAG_OLLAMA_URL", "http://127.0.0.1:11434")
            timeout = int(os.environ.get("GNN_RAG_OLLAMA_TIMEOUT", "300"))
            selection = OllamaClient(model=model, endpoint=endpoint, timeout=timeout).select_evidence(
                retrieval["prompt"], tuple(f.evidence_id for f in facts)
            )
            # Unknown IDs, invalid JSON and no evidence all fail closed.
            response = render_verified_answer(facts, selection.ids)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"answer": response, "model": model,
                "selected_evidence_ids": list(selection.ids),
                "selection_status": selection.status,
                "catalog": catalog, "evidence": retrieval["evidence"],
                "note": "Facts are generated by a fixed, verified KG renderer. LLM selects IDs only. Not medical advice."}

    def _execute_comparison(request: RetrievalRequest, progress=None) -> dict:
        """Run a real four-mode qualitative comparison and persist its PDF.

        Plain Qwen sees only question and node label. Other three modes use
        the SAME training graph but different rankings; model selects verified
        IDs, never invents prose. Held-out link metrics do not evaluate plain Qwen.
        """
        from .grounded import EvidenceFact
        if progress: progress('loading_graph')
        index, table = resources()
        ranker = relation_ranker()
        if request.node_id not in index.node_to_row:
            raise HTTPException(status_code=404, detail="Unknown start node")
        if request.relation is not None and request.relation not in ranker.rel_to_row:
            raise HTTPException(status_code=422, detail="Unknown relation code")
        observed = directory / "train_edges.csv"
        if not observed.is_file():
            raise HTTPException(status_code=503, detail="Run training first")
        known = pl.read_csv(observed)
        node_data = table.filter(pl.col('node_id') == request.node_id).to_dicts()
        node_name = node_data[0]['name'] if node_data else request.node_id
        alpha, calibrated = .5, False
        evaluation_path = directory / 'evaluation.json'
        if evaluation_path.is_file():
            try:
                v = json.loads(evaluation_path.read_text(encoding='utf-8'))
                a = float(v['calibration']['alpha'])
                if np.isfinite(a) and 0. <= a <= 1. and not v['calibration'].get('uses_test_labels', False):
                    alpha, calibrated = a, True
            except (OSError, KeyError, TypeError, ValueError):
                pass
        client = OllamaClient(model=os.environ.get('GNN_RAG_OLLAMA_MODEL', 'qwen3:1.7b'),
                              endpoint=os.environ.get('GNN_RAG_OLLAMA_URL', 'http://127.0.0.1:11434'),
                              timeout=int(os.environ.get('GNN_RAG_OLLAMA_TIMEOUT', '300')))
        result = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'node_id': request.node_id, 'node_name': node_name,
            'question': request.question, 'hops': request.hops,
            'dataset': json.loads((directory / 'run_summary.json').read_text(encoding='utf-8'))['dataset'],
            'top_k': request.top_k, 'model': client.model,
            'fusion_alpha': alpha, 'fusion_calibrated': calibrated,
            'results': {},
            'intermediate_representation_version': 'v14-typed-paths-structured-llm',
            'disclaimer': ('Plain Qwen does not receive KG context. The other three methods '
                           'only use observed train edges. This is qualitative QA comparison, '
                           'not a four-way MRR/medical correctness benchmark.'),
        }
        # Evaluate the raw LLM first, without any graph-derived information.
        if progress: progress('plain_qwen')
        try:
            plain = client.plain_answer(node_name, request.question)
            result['results']['plain_qwen'] = {
                'answer': plain, 'status': 'unverified_generation',
                'catalog': [], 'evidence': [], 'selected_evidence_ids': [],
            }
        except (RuntimeError, ValueError) as exc:
            result['results']['plain_qwen'] = {
                'answer': '', 'status': 'error', 'error': str(exc),
                'catalog': [], 'evidence': [], 'selected_evidence_ids': [],
            }
        for method in ('bfs_only', 'gnn_only', 'bfs_gnn'):
            if progress: progress(method)
            paths = grounded_search(
                index, known, request.node_id, hops=request.hops,
                top_k=request.top_k, beam_width=request.beam_width,
                edge_ranker=ranker, relation_filter=request.relation,
                ranking_method=method, fusion_alpha=alpha,
            )
            facts = evidence_catalog(table, paths)
            catalog = [asdict(f) for f in facts]
            evidence_paths = [{'nodes': p.nodes, 'distance': p.distance,
                               'edges': [vars(e) for e in p.edges]} for p in paths]
            if not facts:
                result['results'][method] = {
                    'answer': render_verified_answer(facts, ()),
                    'status': 'insufficient', 'catalog': catalog,
                    'evidence': evidence_paths, 'selected_evidence_ids': [],
                }
                continue
            try:
                selection = client.select_evidence(
                    evidence_prompt(table, paths, request.question),
                    tuple(f.evidence_id for f in facts))
                answer_text = render_verified_answer(facts, selection.ids)
                result['results'][method] = {
                    'answer': answer_text, 'status': selection.status,
                    'selected_evidence_ids': list(selection.ids),
                    'catalog': catalog, 'evidence': evidence_paths,
                }
            except (RuntimeError, ValueError) as exc:
                result['results'][method] = {
                    'answer': '', 'status': 'error', 'error': str(exc),
                    'catalog': catalog, 'evidence': evidence_paths,
                    'selected_evidence_ids': [],
                }
        # v11: first independently interpret each node/edge sequence (clearly
        # marking hypotheses), then compare all four FINISHED results in Qwen.
        # Neither generative step changes the verified original answers.
        add_explanations(result, client,
                         on_update=(lambda method: progress('explain_' + method)) if progress else None)
        if progress: progress('cross_comparison')
        add_cross_comparison(result, client)
        if progress: progress('saving_report')
        location = Path(os.environ.get('GNN_RAG_COMPARISON_DIR', '/app/outputs/comparison'))
        try:
            location.mkdir(parents=True, exist_ok=True)
            generate_four_way_pdf(location / FOUR_WAY_PDF, result)
            pending = location / (FOUR_WAY_JSON + '.tmp')
            pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            pending.replace(location / FOUR_WAY_JSON)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=f'Comparison completed but PDF saving failed: {exc}') from exc
        return {**result, 'json_url': '/api/comparison-json', 'pdf_url': '/api/comparison-report',
                'pdf_name': FOUR_WAY_PDF}

    @api.post('/compare-answers')
    def compare_answers(request: RetrievalRequest) -> dict:
        """Legacy synchronous endpoint; React v12 uses /compare-jobs to avoid timeouts."""
        return _execute_comparison(request)

    @api.post('/compare-jobs', status_code=202)
    def start_comparison_job(request: RetrievalRequest) -> dict:
        """Return immediately. Poll /compare-jobs/{job_id} while Ollama runs."""
        try:
            job_id = jobs.submit(lambda progress: _execute_comparison(request, progress))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {'job_id': job_id, 'status': 'queued'}

    @api.get('/compare-jobs/{job_id}')
    def comparison_job_status(job_id: str) -> dict:
        status = jobs.status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail='Comparison job not found (the API may have restarted)')
        return status

    @api.get('/comparison-report')
    def comparison_report():
        location = Path(os.environ.get('GNN_RAG_COMPARISON_DIR', '/app/outputs/comparison'))
        target = location / FOUR_WAY_PDF
        if not target.is_file():
            raise HTTPException(status_code=404, detail='Run Compare Four Methods first')
        return FileResponse(target, media_type='application/pdf', filename=FOUR_WAY_PDF)

    @api.get('/comparison-json')
    def comparison_json():
        """Diagnostic JSON includes bounded raw model responses on failure."""
        location = Path(os.environ.get('GNN_RAG_COMPARISON_DIR', '/app/outputs/comparison'))
        target = location / FOUR_WAY_JSON
        if not target.is_file():
            raise HTTPException(status_code=404, detail='Run Compare Four Methods first')
        return FileResponse(target, media_type='application/json', filename=FOUR_WAY_JSON)

    @api.post("/predict-triples")
    def predict_triples(request: TripleQuery) -> dict:
        """Model hypotheses, NEVER factual evidence and NEVER medical advice."""
        try:
            ranker = relation_ranker()
            if request.node_id not in ranker.id_to_row or request.relation not in ranker.rel_to_row:
                raise HTTPException(status_code=404, detail='Unknown node or relation')
            head = ranker.id_to_row[request.node_id]
            relation_id = ranker.rel_to_row[request.relation]
            spec = RELATIONS.get(request.relation)
            if spec is not None and ranker.kinds[head] != spec.source_kind:
                raise HTTPException(status_code=422, detail=(
                    f'Relation {request.relation} requires source kind {spec.source_kind}; '
                    f'got {ranker.kinds[head]}'))
            required_kind = spec.target_kind if spec is not None else None
            allowed = [i for i, kind in enumerate(ranker.kinds)
                       if i != head and (required_kind is None or kind == required_kind)]
            observed = pl.read_csv(directory / 'train_edges.csv')
            known = {r['target'] for r in observed.iter_rows(named=True)
                     if r['source'] == request.node_id and r['relation'] == request.relation}
            if spec is not None and not spec.directed and spec.source_kind == spec.target_kind:
                known.update(r['source'] for r in observed.iter_rows(named=True)
                             if r['target'] == request.node_id and r['relation'] == request.relation)
            allowed = np.asarray([i for i in allowed if ranker.ids[i] not in known], dtype=np.int32)
            summary = json.loads((directory / 'evaluation.json').read_text(encoding='utf-8'))
            alpha = float(summary['calibration']['alpha'])
            rows = pl.read_csv(directory / 'train_edges.csv').select('source', 'relation', 'target').iter_rows(named=True)
            train_triples = np.asarray([(ranker.id_to_row[r['source']],
                                         ranker.rel_to_row[r['relation']],
                                         ranker.id_to_row[r['target']]) for r in rows], dtype=np.int32).reshape(-1, 3)
            predictions = rank_candidates(ranker, train_triples, head, relation_id, allowed,
                                          alpha=alpha, method=request.method)[:request.top_k]
            return {'head': request.node_id, 'relation': request.relation,
                    'candidate_target_kind': required_kind, 'method': request.method,
                    'calibrated_gnn_weight': alpha,
                    'predictions': [{'target': ranker.ids[i],
                                     'gnn_score': ranker.edge_score(head, relation_id, i)}
                                    for i in predictions],
                    'warning': 'Unobserved links are model hypotheses, NOT verified medical facts. The displayed gnn_score is NOT the blended ranking score.'}
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @api.post("/retrieve")
    def retrieve(request: RetrievalRequest) -> dict:
        try:
            index, table = resources()
            paths = index.multi_hop(
                request.node_id, hops=request.hops,
                top_k=request.top_k, beam_width=request.beam_width,
            )
            context = make_context(table, paths, question=request.question)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=503, detail="Prepare demo artifacts first") from exc
        return {
            "paths": [{"node_ids": p.node_ids, "distances": p.distances,
                       "score": p.score} for p in paths],
            "context": context,
            "note": "ANN hop chains are not verified KG-edge paths; no LLM call occurs",
        }

    return api


app = create_app()
