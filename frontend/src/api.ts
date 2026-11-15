export type Health = {ok:boolean;artifact_directory:string;dataset?:string|null;expected_dataset?:string|null;dataset_matches?:boolean}
export type OllamaStatus = {reachable:boolean;model_present:boolean;model:string;endpoint?:string;note?:string;error?:string}
export type Info = {dataset:string;start_id:string;observed_edges:number;triple_aware?:boolean;relations:string[];model:string}
export type Node = {node_id:string;name:string;topic:string}
export type Relation = {code:string;source_kind:string|null;target_kind:string|null;label:string}
export type Score = {mean:number;sd:number|null;per_seed:number[]}
export type ScaleSummary = {size:number;num_seeds:number;seeds:number[];mean_node_count:number;mean_candidate_count:number;methods:Record<string,Record<string,Score>>;alphas:number[];training?:Record<string,number>}
export type Benchmark = {parameters:{created_utc:string;feature_mode:string;epochs:number;seeds:number[];max_positives_per_epoch:number;eval_max_queries:number};sizes:Record<string,ScaleSummary>}
export type EvidenceFact = {evidence_id:string;source:string;target:string;source_name:string;target_name:string;source_kind:string;target_kind:string;relation:string;label:string;provenance:string;statement:string}
export type EvidencePath = {nodes:string[];distance:number;edges:{source:string;target:string;relation:string}[]}
export type EvidenceResponse = {evidence:EvidencePath[];catalog:EvidenceFact[];note:string}
export type AnswerResponse = {answer:string;selection_status:string;selected_evidence_ids:string[];model:string;catalog:EvidenceFact[];evidence:EvidencePath[]}
export type VerifiedStep = {hop:number;evidence_id:string;source_id:string;source_name:string;source_kind:string;relation_code:string;relation_label:string;provenance:string;target_id:string;target_name:string;target_kind:string;traversal_from:string;traversal_to:string;traversal_direction:'forward'|'reverse';selected_for_answer:boolean;observed_train_edge:true}
export type VerifiedPath = {path_id:string;traversal_node_ids:string[];steps:VerifiedStep[];status:'verified_observed_train_path'}
export type MethodAudit = {source:'no_kg_retrieval'|'observed_train_kg';selected_train_edges:EvidenceFact[];path_count:number;note:string}
export type FourWayEntry = {answer:string;status:string;error?:string;selected_evidence_ids:string[];catalog:EvidenceFact[];evidence:EvidencePath[];structured_paths?:VerifiedPath[];structured_path_text?:string;interpretation_warning?:string;sequence_interpretation:string;explanation:string;explanation_status:'generated_unverified'|'error'|'skipped_no_answer';explanation_error?:string;generation_warning?:string;explanation_attempts?:string[]}
export type FourWayComparison = {created_utc:string;dataset?:string;node_id:string;node_name:string;question:string;hops:number;top_k:number;model:string;fusion_alpha:number;fusion_calibrated:boolean;results:Record<'plain_qwen'|'bfs_only'|'gnn_only'|'bfs_gnn',FourWayEntry>;pdf_url:string;pdf_name:string;disclaimer:string;explanation_note:string;cross_factual_comparison?:{plain_qwen_kg_edges:number;selected_counts:Record<string,number>;common_triples:string[][];unique_triples:Record<string,string[][]>;summary_text:string};json_url?:string;method_audit?:Record<'plain_qwen'|'bfs_only'|'gnn_only'|'bfs_gnn',MethodAudit>;intermediate_representation_version?:string;cross_comparison:string;cross_comparison_status:'generated_unverified'|'error'|'skipped_no_answers'|'not_run';cross_comparison_error?:string;cross_comparison_note?:string}
export type Prediction = {target:string;gnn_score:number}
export type PredictionResponse = {head:string;relation:string;method:string;calibrated_gnn_weight:number;predictions:Prediction[];warning:string}
export type Evaluation = {methods?:Record<string,{test?:Record<string,number>}>;calibration?:{alpha?:number};query_count?:number}

export type ComparisonJob = {job_id:string;status:'queued'|'running'|'completed'|'failed';stage?:string;error?:string;result?:FourWayComparison}
export async function request<T>(path:string, init?:RequestInit):Promise<T> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), path === '/answer' || path === '/compare-answers' ? 1050000 : 30000)
  try {
    const res = await fetch('/api'+path, { ...init, signal:controller.signal, headers: { ...(init?.body?{'Content-Type':'application/json'}:{}), ...init?.headers } })
    if (!res.ok) {
      const body = await res.json().catch(()=>null) as {detail?:string}|null
      throw new Error(body?.detail ?? `HTTP ${res.status}: ${res.statusText}`)
    }
    return await res.json() as T
  } finally { clearTimeout(timeout) }
}
export function post<T>(path:string,payload:unknown):Promise<T> {return request<T>(path,{method:'POST',body:JSON.stringify(payload)})}
