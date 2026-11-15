# v4 アルゴリズム改修と検証記録（2026-11-14）

## 背景・観測された問題

以前のユーザー実機での v3.1 120ノード級Hetionet部分グラフでは、保留リンクを全ノードから検索する実験で次を観測しました。

| 手法 | Hits@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| `gnn_voyager` | 0.268 | 0.201 | 0.046 |
| `graph_bfs` | 0.464 | 0.350 | 0.255 |
| `random` | 0.133 | 0.085 | 0.041 |

この結果は**v3.1の関係非依存・全ノード候補検索**で得られた値です。v4は関係条件付きTriple検索へ問題定義を変更したため、この数字との直接比較はできません。

### 旧設計のズレ

- SIFの `(head, relation, tail)` をノードペアに潰し、異なる関係ラベルを一部捨てていた。
- 学習時には任意の未接続ノードを負例とし、評価時にはほぼ全ノードから検索していた。
- 検索クエリは関係やtarget kindを持たない単なる近接探索だった。
- RAGの既知の根拠（Observed KG）と、未知リンクの予測（Link Prediction）は別タスクである。

## v4の設計

1. **PolarsによるTriple保持**: `source, relation, target, weight`。重複の定義は3要素完全一致。同じsource/targetで異なるrelationは保持。Converterは `schema_version=4` を記録する。
2. **漏洩を避ける分割**: 同一の無向ノードペアに属するすべてのTripleは同じpartition。Message passingには訓練エッジ以外を入れず、複数関係で同じペアの重みが増えないようユニーク化。
3. **ノード特徴**: source nameからSHA-256で固定hash化した文字3-gram（64バケット）と `kind`、`topic` がkindと異なる場合だけtopicを使用。外部言語モデルや保留リンクから特徴を生成しない。
4. **モデル**: JAX GraphSAGE-style encoder（2層）＋ relation query vector。関係の意味を各エッジのメッセージ自体に埋め込むR-GCNではない。
5. **ネガティブサンプリング**: 例 `(Disease, DaG, Gene)` なら**Geneだけ**から取る。`(head, relation)` の訓練で観測済みの正例は除く。test/validation正例は訓練時のサンプリング判定に使わない。
6. **検索に近い目的関数**: typed sampled-softmax を使用して、観測されたtargetが同一関係・同一型の負例群より上位になるよう最適化。
7. **既知エッジのKG-RAG検索**: 実際の訓練エッジの隣接から候補を作り、関係スコアで再ランキング。存在しない潜在ベクトル近傍を根拠としてLLMへ渡さない。
8. **未知Triple予測**: `(head, relation)` → target kind を固定。厳密スコアとVoyager（target kindごとのHNSW）を比較する。ここで出る未観測Tripleは**医学的事実ではない**。

### 数式

グラフの学習済みノード埋め込みと関係ベクトルを

\[
  z_v = f_\theta(X, E_\mathrm{train})_v,\quad u_r\in\mathbb R^d
\]

とし、関係条件付きクエリと候補スコアを

\[
  q_{h,r}= \frac{z_h+u_r}{\|z_h+u_r\|_2}, \qquad
  s(h,r,t)=\tau\,q_{h,r}^{\top}z_t, \qquad
  \tau = \exp(\mathrm{clip}(\gamma,-2,3))
\]

とします。関係ごとの候補集合は

\[
  C(h,r)=\{t: \mathrm{kind}(t)=\mathrm{targetKind}(r),\ t\neq h\}
\]

です。未型付けの合成関係 `related_to` では候補型制約を適用しません。訓練時はtrain正例を除いて負例をサンプリングし、次を最小化します。

\[
\mathcal L = -\frac1N \sum_{(h,r,t^+)\in\mathcal T_{\mathrm{train}}}
\log \frac{e^{s(h,r,t^+)}}{e^{s(h,r,t^+)}+\sum_{t^-\in \mathcal N(h,r)}e^{s(h,r,t^-)}}
\]

ただし型制約下で有効な負例がないTripleは損失から除外します。計算上、JAXのGraphSAGE encoderはtrain adjacencyにのみ依存し、relation埋め込みはquery scoringに使用します。

### 向きの取り扱い

- `Gr>G` など `directed=True` は逆向きとして扱わない。
- `DrD`, `GiG` 等 `source_kind == target_kind` かつ `directed=False` と登録された関係は、**訓練の正例と評価クエリに逆向きも追加**。記録CSV上は元のTripleの向きが残る。
- cross-kind関係ではユーザーの推測による逆向きエッジを足さない。

## 比較実験の条件

比較する方法：`gnn_exact`、`gnn_voyager`、`gnn_no_relation`、`graph_bfs`、`random`。`gnn_no_relation`は同一v4埋め込みに対するクエリの関係ベクトル無効化アブレーションです（旧v3.1モデルを再学習した結果ではありません）。

- テストTriple `(head, relation, tail)` について `(head, relation)` をクエリとし、正しい型のtailだけを候補にする。
- trainの既知同関係tailは除外する。validationの正例は**評価計算時に限り**除外し、未正解と誤カウントしない（filtered evaluation）。これはtrainの損失・候補サンプルには用いない。
- 一つのクエリに複数test tailがある場合、それらを一つの集合としてHits@K・Recall@K・MRR@Kを求める。
- BFSは訓練時だけの無向隣接に基づく最短距離・次数によるTie-break。Randomは同じ候補集合で既定20回。
- Voyagerは候補target-kindごとのインデックスに relation query を渡し、HNSWの結果を**同じ厳密スコアで再ランキング**。K件に足りないときだけ過剰取得する。
- 記録先 `evaluation.md`, `evaluation.json`, `evaluation_per_query.csv`。評価基準や候補集合が異なる旧v3.1数値との直接の優劣を主張しない。

## 検証実績と限界

今回の作成環境では、以下を実行しました。

- `tests/test_v4_triple_core.py`: 5件。Group split漏洩チェック・再現性・型に合う負例・名称特徴・JAX学習/勾配と検索ランキング。
- `tests/test_v4_adapters_offline.py`: 3件。複数relationの保存、旧キャッシュ再サンプリング判定、Voyager互換のテスト用Indexに対する候補順位検証。
- 他のオフライン回帰テストと合わせて **48件成功**。
- Python AST、Compose YAMLの依存順序（`prepare-data -> train -> api`）を静的検証。

**未検証:** 作成環境には `polars`、`voyager`、Dockerエンジンがないため、実際のHetionetデータを使ったv4のDocker統合実行、Voyager C++実装、Qwen3での生成、旧モデルとの公正な同一条件アブレーションは実施できていません。Macで `docker compose --profile test run --build --rm test` と `docker compose up --build -d --force-recreate` を実行し、`evaluation.md` を確認してください。

### 新版で最初に見るべき情報

```bash
docker compose logs --no-color --tail=120 prepare-data train
cat data/hetionet_demo/dataset_meta.json
cat outputs/local_rag/run_summary.json
cat outputs/local_rag/evaluation.md
```

`dataset_meta.json` に `schema_version: 4`、`triple_preserving: true` があり、`run_summary.json` に `triple_aware: true`、`metrics.training_objective: sampled softmax ranking ...` が出ることを確認してください。

## 研究上の限界（面接で伝えること）

1. 部分グラフはAlzheimer's disease中心の選択バイアスを持つ。関係ごとのテスト分布も均一ではない。
2. name hashは語彙の意味表現でなく、単純な文字列特徴。異種グラフの関係情報を完全に取り込むわけではない。
3. sampled-softmaxは未観測をnegativeとして扱う近似。Incomplete KGは未観測正例を含みうる。
4. Voyagerは小規模KGでは近似の性能・速度に意味が小さい。ここでは**計算の整合性比較**を行う。
5. Link predictionのMRRが高くても、医学的な真実性・回答品質は保証できない。
6. 複数seed、relation別のRecall、同じtyped candidate protocolでの旧GNNデコーダーのアブレーションを実施して初めて、各変更の寄与を議論できる。
