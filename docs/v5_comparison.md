# v5: 同じ候補集合で GNN / BFS / BFS+GNN を比較（2026-11-14）

## 狙いとスコープ

- 旧v3.1で観測された、GNN+Voyager の Hits@10=0.268 / MRR@10=0.046 と BFS の Hits@10=0.464 / MRR@10=0.255 は**関係なしの全ノード検索課題**だった。
- v4/v5では `(head, relation) -> tail` の**型制約付き保留Triple発見**へ問題設定を変更。v3.1の数字との直接比較は不可。
- v5は **v4と同じ学習済みGraphSAGE + relation decoder** をそのまま使い、**ランキング関数だけ変更**した。モデル自体の改良効果と混同しない。

## 評価設定

1. `(source, relation, target)` を保持する。複数関係を持つ同じ無向ノードペアは同じtrain/validation/test partitionへ入れる。
2. GNNのメッセージパッシング・負例・BFS隣接は**trainだけ**を使う。
3. モデルパラメータ学習にはvalidation/test正例を使わない。**融合重みはvalidation上のMRR@10だけで選択**する。testを使って微調整しない。
4. 3手法は、クエリ `(head, relation)` ごとに **同じ候補ターゲット、同じ評価正例、同じK** を使う。候補はrelationのtarget型に限定し、既知train正例を除外。test時のみ他のvalidation正例をfiltered evaluationで除外する（評価時の常識的な扱いで、trainや重み選択に使わない）。
5. BFSはtrain無向隣接の最短ホップ数、tie-breakは次数降順→node index昇順。複数経路や関係型による距離重み付けは含まない（旧BFSと同一条件）。
6. `gnn_only` は**全候補に対する厳密スコア**で順位付け。近似Voyagerの上位K絞り込みは入れず、ANN精度の影響を隔離する。
7. `bfs_gnn` は双方の**順位percentile**を線形融合する。cosineと最短距離という異尺度の生スコアを直接混ぜない。

### スコア

\[
  s_\mathrm{GNN}(h,r,t)=\left\langle\frac{z_h+u_r}{\|z_h+u_r\|_2},z_t\right\rangle
\]

\[
  p_i(t) = 1 - \frac{\operatorname{rank}_i(t)}{\max(|C|-1,1)}
\]

\[
  \boxed{s_\alpha(t)=\alpha p_\mathrm{GNN}(t)+(1-\alpha)p_\mathrm{BFS}(t)}
\]

- α∈{0,0.05,...,1} の中からvalidation macro-MRR@10最大を選ぶ。MRR同点ならHits@1最大、さらに同点ならαの小さいものを選ぶ。
- `alpha=0` ではBFSと全く同じ出力になることを単体テストで検証。`alpha=1` はGNNと同じ。
- `alpha=0` を選んだ場合は「融合による改善」とは呼べない。`alpha≈1` で数値が同じなら「GNNと同水準」と判断する。

## 実測: **合成**108ノード・468Triple、3 seed、25 epoch

オフライン実行環境にはPolars / Voyager / DockerおよびユーザーMac上のHetionetデータが存在しない。そこで実データの数値を創作せず、JAXモデルと順位融合の実装・数値計算を検証するため、**6クラスターの人工的なノード名・関係をもつ合成グラフ**を使った。クラスター名はノード特徴量に含まれ、意味的にGNN側に有利な設定であり、一般化はできない。

| 手法 | 平均Hits@1 | 平均Hits@10 | 平均MRR@10 | 平均Recall@10 |
|---|---:|---:|---:|---:|
| GNN単独 | 0.1607 | 0.9826 | **0.3847** | 0.9714 |
| BFS単独 | 0.0308 | 0.4427 | 0.1262 | 0.4034 |
| BFS+GNN | 0.1607 | 0.9826 | **0.3847** | 0.9714 |

| Seed | validation選択α | GNN MRR@10 | BFS MRR@10 | Hybrid MRR@10 |
|---|---:|---:|---:|---:|
| 41 | 0.95 | 0.3473 | 0.1000 | 0.3473 |
| 42 | 0.95 | 0.4080 | 0.1310 | 0.4080 |
| 43 | 0.95 | 0.3986 | 0.1475 | 0.3986 |

**結論（合成グラフのみ）:** GNNはこの人工条件のBFSを上回った。一方、**BFS+GNNはGNN単独より良くなっていない**。α=0.95となり、ほぼGNN順位のまま。Hetionetでの改善効果は未検証。差の有意性も検証していない。

実測のソースは `examples/offline_benchmark.json`、再実行スクリプトは `scripts/offline_compare.py`。

## 検証内容

- v4既存のJAX学習・ノード対単位の分割・負例型制約に加え、v5で **(a) α=0/1 の完全再現、(b) 3手法が同じ候補集合を使うこと、(c) TEST labels を変えてもαが不変、(d) BFSをtrain graphのみから計算、(e) APIの手法別ランキングが評価ロジックと一致** をチェック。
- 単体テストはPolars/Voyagerに依存しない環境で実施。JAXは実際にCPU上で学習させ、形式だけのモック数値を使っていない。
- 本環境にはHetionet本体もDockerエンジンもないため、**Hetionetでのv5精度、VoyagerのC++、Ollama、Compose実起動についての成功は主張しない**。

## Macで実際のHetionetを測る

```bash
cp .env.example .env
# .envで GNN_RAG_DATASET=hetionet を確認
# 旧バージョンと混在させず、新規のoutputsを使う
docker compose --profile test run --build --rm test
docker compose up --build -d --force-recreate
cat outputs/local_rag/evaluation.md
```

出力された `evaluation.md` に3手法のHits@K / Recall@K / MRR@K が並ぶ。`evaluation.json` の calibration.alpha と候補数の分布も同時に確認する。旧v3.1や合成実験の数値をHetionetの実測として扱わない。

## 複数seedでのHetionet実測コマンド（任意）

```bash
docker compose --profile benchmark run --build --rm benchmark
cat outputs/multiseed/summary.md
```

各seedのtrain/validation/test分割と学習を独立に行い、**validationのみ**でαを校正。3手法のtest MRR@10の平均・標準偏差と、同じseed内でのペア差（BFS+GNN − BFS、BFS+GNN − GNN）を保存。通常APIの `outputs/local_rag` は変更しない。
