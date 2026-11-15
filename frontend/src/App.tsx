import { useCallback, useEffect, useMemo, useState } from 'react'
import { request, post, type Health, type Info, type Node, type Relation, type Benchmark, type EvidenceResponse, type AnswerResponse, type PredictionResponse, type Evaluation, type ScaleSummary, type FourWayComparison, type ComparisonJob, type OllamaStatus } from './api'

type Page = 'overview' | 'explore' | 'predict' | 'benchmark'
type Method = 'gnn_only' | 'bfs_only' | 'bfs_gnn'
const methods: {id:Method; name:string; short:string; color:string}[] = [
  {id:'gnn_only',name:'GNN only',short:'GNN',color:'#3977b8'},
  {id:'bfs_only',name:'BFS only',short:'BFS',color:'#a0afc3'},
  {id:'bfs_gnn',name:'BFS + GNN',short:'Fusion',color:'#13a394'},
]
const fourModes = [{id:'plain_qwen',name:'Qwen only',note:'KGを参照しない生成回答'}, {id:'bfs_only',name:'BFS only',note:'Trainグラフ構造による検索'}, {id:'gnn_only',name:'GNN only',note:'学習済み埋め込みによる順位'}, {id:'bfs_gnn',name:'BFS + GNN',note:'Validationで調整した順位融合'}] as const
const nav: {id:Page; name:string; subtitle:string; glyph:string}[] = [
  {id:'overview',name:'Overview',subtitle:'システム状況',glyph:'◫'},
  {id:'explore',name:'Graph Explorer',subtitle:'KG根拠の検索',glyph:'⌘'},
  {id:'predict',name:'Link Prediction',subtitle:'関係付き候補予測',glyph:'⌁'},
  {id:'benchmark',name:'Benchmarks',subtitle:'10k / 100k 比較',glyph:'▥'},
]
const fmt = (n:number | null | undefined, digits=4) => typeof n==='number' && Number.isFinite(n) ? n.toFixed(digits) : '—'
const num = (n:number | undefined) => typeof n==='number' ? n.toLocaleString('ja-JP') : '—'
const sd = (v:{mean:number;sd:number|null}|undefined) => v ? `${fmt(v.mean)}${v.sd == null ? '' : ` ± ${fmt(v.sd)}`}` : '—'
const root = 'Hetionet_10k_100k_comparison.pdf'

function Pill({ok,children}:{ok:boolean;children:string}) { return <span className={`status-pill ${ok?'good':'bad'}`}><i />{children}</span> }
function Metric({label,value,caption,accent}:{label:string;value:string;caption?:string;accent?:boolean}) {return <div className={`metric ${accent?'accent':''}`}><div className="metric-label">{label}</div><div className="metric-value">{value}</div>{caption&&<div className="metric-caption">{caption}</div>}</div>}
function Empty({title,description}:{title:string;description:string}) {return <div className="empty"><div className="empty-icon">◌</div><strong>{title}</strong><p>{description}</p></div>}
function ScoreChart({data}:{data:ScaleSummary}) {
  const max=Math.max(.01,...methods.map(m=>data.methods[m.id]?.['mrr@10']?.mean ?? 0))*1.18
  return <div className="bar-chart">{methods.map(m=> {const score=data.methods[m.id]?.['mrr@10'];return <div className="bar-row" key={m.id}><div className="bar-name">{m.name}</div><div className="bar-track"><div className="bar-fill" style={{width:`${100*(score?.mean ?? 0)/max}%`,background:m.color}} /></div><span className="bar-num">{fmt(score?.mean)}</span></div>})}</div>
}
function EvidenceGraph({paths,names}:{paths:EvidenceResponse['evidence'];names:Map<string,Node>}) {
  const unique=new Map<string,{source:string;target:string;relation:string}>()
  for (const path of paths) for (const edge of path.edges) {
    const key=`${edge.source}|${edge.target}|${edge.relation}`
    if(!unique.has(key))unique.set(key,edge)
  }
  const edges=Array.from(unique.values()).slice(0,24)
  const ids=Array.from(new Set(edges.flatMap(e=>[e.source,e.target]))).slice(0,18)
  if(ids.length<2)return null
  const width=700,height=246,rx=width*.34,ry=height*.32
  const positions=new Map(ids.map((id,i)=>[id,{
    x:width/2+rx*Math.cos((i/ids.length)*Math.PI*2-Math.PI/2),
    y:height/2+ry*Math.sin((i/ids.length)*Math.PI*2-Math.PI/2),
  }]))
  return <div className="graph-box"><div className="graph-box-label">EVIDENCE PATH GRAPH <small>Train edges only · {edges.length} links</small></div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="観測済みエッジのグラフ図">
      <defs><marker id="edgeArrow" viewBox="0 0 10 10" refX="13" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9db8c7"/></marker></defs>
      {edges.map((e,i)=>{const a=positions.get(e.source),b=positions.get(e.target);return a&&b?<line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#aec6d3" strokeWidth="1.25" strokeOpacity=".72" markerEnd="url(#edgeArrow)"/>:null})}
      {ids.map((id,i)=>{const p=positions.get(id)!,node=names.get(id),label=node?.name||id;return <g key={id} transform={`translate(${p.x},${p.y})`}><circle r="13" fill={i===0?'#139b93':'#32739a'} stroke="#fff" strokeWidth="2.5"/><text y="4" textAnchor="middle" fill="white" fontSize="9" fontWeight="bold">{i+1}</text><text y="28" textAnchor="middle" fontSize="9" fill="#3a5266">{label.length>15?label.slice(0,14)+'…':label}</text><title>{label} · {node?.topic||''}</title></g>})}
    </svg></div>
}

export default function App(){
  const [page,setPage]=useState<Page>('overview')
  const [health,setHealth]=useState<Health|null>(null)
  const [info,setInfo]=useState<Info|null>(null)
  const [ollamaStatus,setOllamaStatus]=useState<OllamaStatus|null>(null)
  const [evaluation,setEvaluation]=useState<Evaluation|null>(null)
  const [benchmark,setBenchmark]=useState<Benchmark|null>(null)
  const [nodes,setNodes]=useState<Node[]>([])
  const [relations,setRelations]=useState<Relation[]>([])
  const [ready,setReady]=useState(false)
  const [fetchError,setFetchError]=useState('')
  const [query,setQuery]=useState('')
  const [nodeId,setNodeId]=useState('')
  const [relation,setRelation]=useState('')
  const [question,setQuestion]=useState('このノードに登録された関係を、根拠だけで説明してください。')
  const [hops,setHops]=useState(2)
  const [topK,setTopK]=useState(3)
  const [method,setMethod]=useState<Method>('bfs_gnn')
  const [evidence,setEvidence]=useState<EvidenceResponse|null>(null)
  const [answer,setAnswer]=useState<AnswerResponse|null>(null)
  const [prediction,setPrediction]=useState<PredictionResponse|null>(null)
  const [fourWay,setFourWay]=useState<FourWayComparison|null>(null)
  const [pending,setPending]=useState('')
  const [compareStage,setCompareStage]=useState('')
  const [actionError,setActionError]=useState('')

  const refresh=useCallback(async()=>{
    setFetchError('')
    const results=await Promise.allSettled([
      request<Health>('/health'),request<Info>('/info'),request<Evaluation>('/evaluation'),
      request<Node[]>('/nodes'),request<Relation[]>('/relations'),request<Benchmark>('/benchmark-summary'),request<OllamaStatus>('/ollama-status')
    ])
    const [h,i,e,n,r,b,o]=results
    if(h.status==='fulfilled')setHealth(h.value); else {setHealth(null);setFetchError('APIを取得できません。Dockerの学習・起動状態を確認してください。')}
    if(i.status==='fulfilled'){setInfo(i.value);setNodeId(prev=>prev||i.value.start_id)}
    if(e.status==='fulfilled')setEvaluation(e.value)
    if(n.status==='fulfilled')setNodes(n.value)
    if(r.status==='fulfilled')setRelations(r.value)
    if(b.status==='fulfilled')setBenchmark(b.value); else setBenchmark(null)
    if(o.status==='fulfilled')setOllamaStatus(o.value); else setOllamaStatus(null)
    setReady(true)
  },[])
  useEffect(()=>{void refresh()},[refresh])

  const nodeMap=useMemo(()=>new Map(nodes.map(n=>[n.node_id,n])),[nodes])
  const selectedNode=nodeMap.get(nodeId)
  const nodeMatches=useMemo(()=>{
    const q=query.trim().toLowerCase();if(!q)return nodes.slice(0,25)
    return nodes.filter(n=>(n.name+' '+n.node_id+' '+n.topic).toLowerCase().includes(q)).slice(0,35)
  },[nodes,query])
  const validRelations=useMemo(()=>relations.filter(r=>!selectedNode?.topic||!r.source_kind||r.source_kind===selectedNode.topic),[relations,selectedNode])
  useEffect(()=>{if(validRelations.length&&!validRelations.some(r=>r.code===relation))setRelation(validRelations[0].code)},[validRelations,relation])
  const scales=useMemo(()=>Object.entries(benchmark?.sizes??{}).sort((a,b)=>Number(a[0])-Number(b[0])),[benchmark])

  const work=async(kind:'evidence'|'answer'|'predict'|'compare')=>{
    if(!nodeId){setActionError('ノードを選択してください。');return}
    setPending(kind);setActionError('')
    try {
      if(kind==='compare'){
        setFourWay(null)
        const payload={node_id:nodeId,question,hops,top_k:topK,beam_width:8}
        setCompareStage('queued')
        const started=await post<ComparisonJob>('/compare-jobs',payload)
        let result:FourWayComparison | undefined
        while(!result){
          await new Promise(resolve=>setTimeout(resolve,2000))
          const state=await request<ComparisonJob>(`/compare-jobs/${started.job_id}`)
          setCompareStage(state.stage||state.status)
          if(state.status==='failed')throw new Error(state.error||'比較ジョブが失敗しました')
          if(state.status==='completed'){
            if(!state.result)throw new Error('比較ジョブの結果がありません')
            result=state.result
          }
        }
        setFourWay(result)
        const graphResult=result.results.bfs_gnn
        setEvidence({catalog:graphResult.catalog,evidence:graphResult.evidence,note:''})
        setAnswer(null)
      } else if(kind==='predict'){
        if(!relation)throw new Error('Relationを選択してください。')
        const result=await post<PredictionResponse>('/predict-triples',{node_id:nodeId,relation,top_k:10,method})
        setPrediction(result)
      } else {
        const payload={node_id:nodeId,question,hops,top_k:topK,beam_width:8}
        if(kind==='evidence'){setEvidence(await post<EvidenceResponse>('/evidence',payload));setAnswer(null)}
        else {const result=await post<AnswerResponse>('/answer',payload);setAnswer(result);setEvidence({catalog:result.catalog,evidence:result.evidence,note:''})}
      }
    } catch(e){setActionError(e instanceof Error?e.message:String(e))}
    finally {setPending('');setCompareStage('')}
  }

  const selectNode=(n:Node)=>{setNodeId(n.node_id);setQuery(n.name);setEvidence(null);setAnswer(null);setPrediction(null);setFourWay(null)}
  const panelTitle=nav.find(n=>n.id===page)!
  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-emblem"><span>G</span></div><div><div className="brand-title">GNN–RAG</div><div className="brand-subtitle">RESEARCH CONSOLE <b>v14</b></div></div></div>
      <div className="sidebar-caption">WORKSPACE</div>
      <nav className="nav">{nav.map(item=><button type="button" key={item.id} onClick={()=>setPage(item.id)} className={`nav-item ${page===item.id?'active':''}`}><span className="nav-glyph">{item.glyph}</span><span className="nav-text"><span>{item.name}</span><small>{item.subtitle}</small></span>{page===item.id&&<span className="nav-mark"/>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="sidebar-card"><div className="sidebar-card-label">DATA SOURCE</div><strong>{info?.dataset === 'hetionet_subset' ? 'Hetionet v1.0' : info?.dataset === 'synthetic' ? 'Synthetic demo graph' : (info?.dataset || 'Dataset: loading')}</strong><small>{info?.dataset === 'hetionet_subset' ? 'Biomedical knowledge graph' : info?.dataset === 'synthetic' ? 'Synthetic KG — not Hetionet' : 'Dataset source pending'}</small><div className="sidebar-rule"/><span>JAX · GraphSAGE · Voyager</span><span>Ollama · Qwen3 1.7B</span></div><div className="sidebar-footer">LOCAL RESEARCH DEMO · NOT CLINICAL ADVICE</div></div>
    </aside>
    <div className="content">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <strong>{panelTitle.name}</strong></div><div className="top-actions"><span className="top-environment">LOCAL ENVIRONMENT</span><Pill ok={Boolean(health?.ok)}>{health?.ok?'API healthy':'API not ready'}</Pill><button className="icon-button" onClick={()=>void refresh()} title="更新" aria-label="更新">↻</button></div></header>
      <main className="main">
        <div className="heading"><div className="eyebrow">GRAPH INTELLIGENCE / RESEARCH</div><h1>{panelTitle.name}</h1><p>{page==='overview'?'異種グラフの学習、根拠検索、モデル比較を一つのコンソールで。':page==='explore'?'観測済みKGエッジに限定した検索と、Qwen3による根拠ID選択。':page==='predict'?'関係条件付きリンク検索。未観測候補は事実ではなくモデル仮説です。':'10k / 100k Tripleで学習・評価した最新の実験結果を可視化。'}</p></div>
        {fetchError&&<div className="notice warning"><strong>接続状況</strong>{fetchError}</div>}
        {!ready&&<div className="loading">データを取得中…</div>}
        {page==='overview'&&<>
          <section className="metrics-grid"><Metric label="DATASET" value={info?.dataset==='hetionet_subset'?'Hetionet subset':info?.dataset==='synthetic'?'Synthetic demo':(info?.dataset||'—')} caption="公開Knowledge Graph"/><Metric label="OBSERVED EDGES" value={num(info?.observed_edges)} caption="Train側で観測したエッジ"/><Metric label="LLM MODEL" value={info?.model||'—'} caption="Ollama · evidence ID selection"/><Metric label="BENCHMARK" value={scales.length===2?'10k / 100k':scales.length?'Partial':'Not run'} caption="規模別評価の状態" accent/></section>
          <div className="two-column">
            <section className="panel"><div className="panel-head"><div><span className="section-kicker">PIPELINE</span><h2>Architecture</h2></div><span className="soft-label">Transductive KG retrieval</span></div><div className="pipeline"><div><span className="step">01</span><strong>Hetionet</strong><small>Node / Relation CSV</small></div><span>→</span><div><span className="step">02</span><strong>JAX GraphSAGE</strong><small>16-d embeddings</small></div><span>→</span><div><span className="step">03</span><strong>Retriever</strong><small>Voyager / verified KG</small></div><span>→</span><div><span className="step">04</span><strong>Ollama</strong><small>Evidence selection</small></div></div><div className="annotation">最終的な医学関連の文は検証されたKGエッジから決定的に生成します。LLMは根拠IDの選択のみ行います。</div>{evaluation?.methods&&<div className="api-eval">CURRENT API / held-out MRR@10 <strong>{fmt(evaluation.methods.bfs_gnn?.test?.['mrr@10'])}</strong><small>API向け単一規模モデルの融合スコア。下の10k/100k studyとは別の実験です。</small></div>}</section>
            <section className="panel"><div className="panel-head"><div><span className="section-kicker">QUICK ACTIONS</span><h2>Explore your model</h2></div></div><div className="actions-list"><button onClick={()=>setPage('explore')}><span className="action-icon">⌘</span><span><strong>根拠経路を確認</strong><small>KG内の観測済みエッジを探索する</small></span><span>↗</span></button><button onClick={()=>setPage('predict')}><span className="action-icon">⌁</span><span><strong>リンク候補を比較</strong><small>GNN / BFS / 融合を切り替える</small></span><span>↗</span></button><button onClick={()=>setPage('benchmark')}><span className="action-icon">▥</span><span><strong>10k / 100k レポート</strong><small>学習比較とPDFを確認する</small></span><span>↗</span></button></div></section>
          </div>
          <section className="panel"><div className="panel-head"><div><span className="section-kicker">LATEST STUDY</span><h2>Cross-scale summary</h2></div><button className="text-button" onClick={()=>setPage('benchmark')}>すべての指標を見る →</button></div>{scales.length ? <div className="summary-grid">{scales.map(([key,data])=><div className="summary-card" key={key}><span className="summary-name">{num(Number(key))} TRIPLES</span><strong>{num(data.mean_node_count)} <small>nodes</small></strong><ScoreChart data={data}/></div>)}</div>:<Empty title="ベンチマーク未実行" description="docker compose --profile study run --build --rm study で10k / 100k学習比較を開始できます。"/>}</section>
        </>}
        {(page==='explore'||page==='predict')&&<div className="work-grid">
          <section className="panel controls-panel">{info?.dataset==='synthetic'&&<div className="notice warning"><strong>合成データで実行中</strong>現在のGraph ExplorerはHetionetではありません。.envのGNN_RAG_DATASET=hetionetを設定して再学習してください。</div>}{ollamaStatus&&!ollamaStatus.reachable&&<div className="notice warning"><strong>Ollama接続不可</strong>{ollamaStatus.error||'MacネイティブOllamaを起動し、host.docker.internal:11434への接続を確認してください。'}</div>}{ollamaStatus?.reachable&&!ollamaStatus.model_present&&<div className="notice warning"><strong>Qwen3が見つかりません</strong>Mac側で ollama pull {ollamaStatus.model} を実行してください。</div>}<div className="panel-head"><div><span className="section-kicker">QUERY PARAMETERS</span><h2>Search setup</h2></div></div>
            <label className="form-label" htmlFor="node-search">Start node</label><input id="node-search" className="field" placeholder="ノード名・ID・型で検索…" value={query} onChange={e=>setQuery(e.target.value)} />
            <div className="node-results" role="listbox" aria-label="候補ノード">{nodeMatches.length?nodeMatches.map(n=><button key={n.node_id} className={n.node_id===nodeId?'selected':''} onClick={()=>selectNode(n)} title={n.node_id}><strong>{n.name}</strong><span>{n.topic} · {n.node_id}</span></button>):<span className="list-empty">該当するノードなし</span>}</div>
            {selectedNode&&<div className="selected-node"><span>SELECTED NODE</span><strong>{selectedNode.name}</strong><code>{selectedNode.node_id}</code></div>}
            {page==='explore'?<>
              <label className="form-label" htmlFor="question">Question</label><textarea id="question" value={question} onChange={e=>setQuestion(e.target.value)} className="field text-area" maxLength={500} rows={4}/>
              <div className="field-pair"><div><label className="form-label" htmlFor="hops">Hops</label><select id="hops" className="field" value={hops} onChange={e=>setHops(+e.target.value)}>{[1,2,3,4].map(x=><option key={x}>{x}</option>)}</select></div><div><label className="form-label" htmlFor="topk">Top K</label><select id="topk" className="field" value={topK} onChange={e=>setTopK(+e.target.value)}>{[1,2,3,5,8,10].map(x=><option key={x}>{x}</option>)}</select></div></div>
              <div className="button-stack"><button disabled={!!pending||!health?.ok} className="primary-button" onClick={()=>void work('evidence')}>{pending==='evidence'?'検索中…':'根拠パスを検索'} <span>→</span></button><button disabled={!!pending||!health?.ok} className="secondary-button" onClick={()=>void work('answer')}>{pending==='answer'?'LLMを処理中…':'Qwen3で回答を構成'}</button><button disabled={!!pending||!health?.ok} className="comparison-button" onClick={()=>void work('compare')}>{pending==='compare'?`比較処理中: ${compareStage||'開始中'}…`:'4方式を解釈・横断比較しPDF保存 →'}</button></div><div className="comparison-help">4方式の原回答からノード・エッジの関係と仮説をそれぞれQwen3で解釈し、続けて3〜6文で説明します。最後に4方式の出力をQwenで横断比較します。解釈・仮説は未検証で、PDFに区別して保存します。</div>
            </>:<>
              <label className="form-label" htmlFor="relation">Relation</label><select className="field" id="relation" value={relation} onChange={e=>setRelation(e.target.value)}>{validRelations.map(r=><option value={r.code} key={r.code}>{r.code} · {r.label}</option>)}</select>
              <label className="form-label" htmlFor="method">Ranking method</label><select className="field" id="method" value={method} onChange={e=>setMethod(e.target.value as Method)}>{methods.map(m=><option key={m.id} value={m.id}>{m.name}</option>)}</select>
              <button disabled={!!pending||!health?.ok} className="primary-button" onClick={()=>void work('predict')}>{pending==='predict'?'予測中…':'候補リンクをランキング'} <span>→</span></button>
            </>}
            {actionError&&<div className="notice error">{actionError}</div>}
          </section>
          <section className="panel results-panel"><div className="panel-head"><div><span className="section-kicker">{page==='explore'?'VERIFIED EVIDENCE':'MODEL HYPOTHESES'}</span><h2>{page==='explore'?'Search results':'Ranked targets'}</h2></div><span className="soft-label">{page==='explore'?'Train edges only':'Unobserved triples'}</span></div>
            {page==='explore'?<>
              {fourWay&&<div className="fourway-area"><div className="fourway-title"><div><span className="section-kicker">FOUR-WAY COMPARISON</span><h3>同じ質問に対する4方式の出力</h3><small>GNN重み α = {fmt(fourWay.fusion_alpha,2)} {fourWay.fusion_calibrated?'（Validation校正済み）':'（暫定値）'}</small></div><a className="download-button" href={fourWay.pdf_url} download={fourWay.pdf_name}>↓ 4方式のPDF</a>{fourWay.json_url&&<a className="download-button" href={fourWay.json_url} download="GNN_RAG_four_way_answers.json">↓ 診断JSON</a>}</div><div className="fourway-grid">{fourModes.map(m=>{const entry=fourWay.results[m.id];return <article className={`fourway-card ${m.id==='plain_qwen'?'unverified':''}`} key={m.id}><header><strong>{m.name}</strong><span>{entry.status}</span></header><small>{m.note}</small><p>{entry.error||entry.answer||'回答なし'}</p>{m.id==='plain_qwen'?<div className="fourway-caution">KG根拠なし。正確性の検証対象外。</div>:<details><summary>検証済みのKG根拠 {entry.catalog.length}件 / 選択 {entry.selected_evidence_ids.join(', ')||'なし'}</summary>{entry.catalog.map(f=><p className="fourway-evidence" key={f.evidence_id}>{f.evidence_id}: {f.statement}</p>)}</details>}<section className="verified-intermediate" aria-label={`${m.name}の確定的な中間表現`}><div className="explanation-heading">確定的な中間表現 <span>{m.id==='plain_qwen'?'KG参照なし':'型・連続性検証済み'}</span></div><pre>{entry.structured_path_text||'中間表現なし'}</pre>{entry.interpretation_warning&&<div className="fourway-caution">{entry.interpretation_warning}</div>}</section><section className="qwen-explanation" aria-label={`${m.name}のQwenによる説明`}>{entry.generation_warning&&<div className="fourway-caution">{entry.generation_warning}</div>}<div className="explanation-heading">Qwenによるシーケンス解釈・説明 <span>{entry.explanation_status==='generated_unverified'?'生成済み・未検証':entry.explanation_status==='error'?'説明生成エラー':'未生成'}</span></div><h4>① ノード・エッジの解釈と想定関係</h4><p>{entry.sequence_interpretation||entry.explanation_error||'シーケンス解釈はありません。'}</p><h4>② 上記生成文の説明（3〜6文）</h4><p>{entry.explanation||entry.explanation_error||'説明は生成されませんでした。'}</p><small>想定される関係は仮説であり、観測済みKGエッジや医学的因果の証明ではありません。Plain QwenにはKGシーケンスを渡しません。</small></section></article>})}</div><section className="verified-intermediate method-audit" aria-label="4方式の根拠監査"><div className="explanation-heading">確定的な根拠照合 <span>Qwenによる生成ではありません</span></div>{fourModes.map(m=>{const audit=fourWay.method_audit?.[m.id];return <div className="audit-method" key={m.id}><strong>{m.name}</strong><small>{m.id==='plain_qwen'?'KG検索なし':`${audit?.selected_train_edges.length||0}件の選択済みTrainエッジ`}</small>{audit?.selected_train_edges.map(f=><p key={f.evidence_id}>[{f.evidence_id}] {f.source_name} ({f.source_kind}) → [{f.relation}: {f.label}] → {f.target_name} ({f.target_kind})<small>出典・定義: {f.provenance}</small></p>)}</div>})}</section><section className="verified-intermediate" aria-label="4方式の確定的な事実比較"><div className="explanation-heading">4方式の確定的な事実比較 <span>Python集計・Qwen生成ではありません</span></div><pre>{fourWay.cross_factual_comparison?.summary_text||'集計なし'}</pre></section><section className="cross-comparison" aria-label="4方式の横断比較"><div className="cross-comparison-header"><span className="section-kicker">FINAL SYNTHESIS</span><h3>③ Qwenによる4方式の横断比較</h3><small>{fourWay.cross_comparison_status==='generated_unverified'?'生成済み・未検証':fourWay.cross_comparison_status==='error'?'生成エラー':'未生成'}</small></div><p>{fourWay.cross_comparison||fourWay.cross_comparison_error||'比較文は生成されていません。'}</p><small>この比較は4方式の生成文と選択済み根拠をもとにした考察です。正確性・因果関係・医療上の妥当性を検証するものではありません。</small></section><div className="notice info">Qwen単独は自由生成、残り3方式の原回答は検証済みKGエッジから決定的に文章化しています。Qwenによるシーケンス解釈・仮説・横断比較はすべて未検証の生成文です。MRR@10などの定量評価とは別です。</div></div>}
              {answer&&<div className="answer"><div className="answer-top"><span className="dot"/> QWEN3 / VERIFIED ANSWER <span className="answer-status">{answer.selection_status}</span></div><p>{answer.answer}</p></div>}
              {evidence?.catalog?.length?<><EvidenceGraph paths={evidence.evidence} names={nodeMap}/><div className="evidence-list">{evidence.catalog.map((f,i)=><div className="evidence-item" key={`${f.evidence_id}-${i}`}><div className="evidence-index">{f.evidence_id}</div><div className="evidence-body"><div className="evidence-link"><strong>{f.source_name}</strong><span className="relationship">{f.relation} · {f.label}</span><strong>{f.target_name}</strong></div><small>{f.source_kind} → {f.target_kind}</small><p>{f.statement}</p></div></div>)}</div></>:<Empty title={evidence?'登録された根拠はありません':'検索結果を表示します'} description="左でノードと質問を指定し、根拠パスを検索してください。"/>}
            </>:prediction?.predictions?.length?<><div className="result-banner"><span>{prediction.method}</span><span>融合のGNN重み α = {fmt(prediction.calibrated_gnn_weight,2)}</span></div><div className="predictions">{prediction.predictions.map((p,i)=><div className="prediction-row" key={p.target}><span className="rank">{String(i+1).padStart(2,'0')}</span><div><strong>{nodeMap.get(p.target)?.name||p.target}</strong><small>{p.target}</small></div><span className="model-score">{fmt(p.gnn_score)}</span></div>)}</div><div className="notice warning"><strong>研究上の注意</strong>{prediction.warning}</div></>:<Empty title="予測結果を表示します" description="左でノード、関係種類、ランキング方式を選び、推論してください。"/>}
          </section>
        </div>}
        {page==='benchmark'&&<>
          <div className="study-banner"><div><span className="study-kicker">REPRODUCIBLE EXPERIMENT</span><h2>Hetionet 10k vs 100k</h2><p>独立に抽出した2規模のグラフでリンクランキング3方式を定量比較。Qwen単独を含む4方式の回答比較はGraph Explorerで実行できます。</p></div><div className="study-actions"><a href="/api/benchmark-report" className={`download-button ${!benchmark?'disabled':''}`} download={root} onClick={e=>{if(!benchmark)e.preventDefault()}}>↓ PDFレポート</a><small>保存先: プロジェクト直下 / {root}</small></div></div>
          {benchmark?<>
            <section className="metrics-grid"><Metric label="TRAINING SEEDS" value={String(benchmark.parameters?.seeds?.length??'—')} caption={benchmark.parameters?.seeds?.join(', ')||'—'}/><Metric label="EPOCHS" value={String(benchmark.parameters?.epochs??'—')} caption="各seedの学習回数"/><Metric label="FEATURE MODE" value={benchmark.parameters?.feature_mode??'—'} caption="Hetionet実ノード属性"/><Metric label="RANKING" value="Exact" caption="Voyager近似は使用しない" accent/></section>
            <div className="two-column">{scales.map(([key,data])=><section className="panel" key={key}><div className="panel-head"><div><span className="section-kicker">{num(Number(key))} TRIPLES</span><h2>{num(data.mean_node_count)} nodes</h2></div><span className="soft-label">{data.num_seeds} seed</span></div><ScoreChart data={data}/><div className="scale-footer"><div><span>候補ノード中央値（平均）</span><strong>{num(Math.round(data.mean_candidate_count))}</strong></div><div><span>最終Train loss（平均）</span><strong>{fmt(data.training?.final_train_loss_mean)}</strong></div></div></section>)}</div>
            <section className="panel"><div className="panel-head"><div><span className="section-kicker">EVALUATION METRICS</span><h2>Held-out link ranking</h2></div><span className="soft-label">mean ± sample SD</span></div><div className="table-scroll"><table><thead><tr><th>Triples</th><th>Method</th><th>MRR@10</th><th>Hits@10</th><th>Recall@10</th></tr></thead><tbody>{scales.flatMap(([key,data])=>methods.map(m=><tr key={`${key}-${m.id}`}><td>{num(Number(key))}</td><td><span className="method-mark" style={{background:m.color}}/>{m.name}</td><td>{sd(data.methods[m.id]?.['mrr@10'])}</td><td>{sd(data.methods[m.id]?.['hits@10'])}</td><td>{sd(data.methods[m.id]?.['recall@10'])}</td></tr>))}</tbody></table></div></section>
            <div className="notice info"><strong>結果の解釈</strong>10kと100kは異なる誘導部分グラフで、候補集合・保留クエリ・関係分布が異なります。規模間のスコアの差は、固定Test条件でのlearning curveではありません。これはリンクランキング評価であり、LLM回答の医学的正確性を示しません。</div>
          </>:<section className="panel"><Empty title="10k / 100k比較は未実行です" description="Macのターミナルで docker compose --profile study run --build --rm study を実行すると、PDFとJSON/CSVが生成されます。"/><pre className="terminal">docker compose --profile study run --build --rm study</pre></section>}
        </>}
      </main><footer className="footer">GNN-RAG v11 <span>•</span> PUBLIC KNOWLEDGE GRAPH RESEARCH <span>•</span> NOT FOR MEDICAL DECISION MAKING</footer>
    </div>
  </div>
}
