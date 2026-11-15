# 旧実装との対応（監査しやすくするための変更点）

対象：提供された `GNN-RAG-main(1).zip`。旧ファイルはユーザーから渡された研究・開発プロトタイプです。今回の提出版は、**現行実装の設計思想を小規模に抽出し、安全に説明できるよう再構成したもの**です。元の訓練結果やパフォーマンスと数値的な同値性はありません。

| 旧ファイル | 役割 | 面接用の置換先 |
|---|---|---|
| `backend/helper_0_1_common_flow.py` | KG読み込みと前処理 | `data.py`, `synthetic.py` |
| `backend/helper_2_integrated_df_src_dst_emb_nodeid.py` | src/dst, node_id, データ結合 | `data.py` |
| `backend/helper_test_model_with_jax.py` | 位置付け: GNN・分割・Link Prediction | `split.py`, `model.py`, `train.py` |
| `backend/helper_1_1_model_with_jax.py` | 旧探索的JAX実装 | `model.py`, `train.py` |
| `backend/helper_3_ann.py` | VoyagerによるANN | `retrieval.py` |
| `backend/helper_test_main_loop.py` | 複数段ANNホップと生成用加工 | `retrieval.py`, `context.py` |
| `backend/helper_8_main_loop.py` | 旧版の全体処理 | `demo.py` |
| `backend/ai_chat.py`, `knowledge_store.py` | Llama.cpp / Marqo連携 | **LLMなし**の`context.py` に縮小 |
| `backend/app.py` | Flask HTTP境界 | オプションの`api.py` (FastAPI) |
| `frontend/` | Next.js/React UI | この小型コードレビュー版では対象外 |

## 特に重要な技術的修正

1. **学習済み埋め込みの取り出し** — `helper_test_model_with_jax.py` の `node_update_fn()` は内部で毎回 `flax_nn.Module` に `mlp.init(jax.random.PRNGKey(0), feats)` を適用し、同ファイルの`train()`が `.nodes` を保存するときも `net_fn(graph).nodes` を使っていました。学習ループの `params` を使った `net.apply(params, ...)` の結果ではありません。改訂版では学習パラメータを`encode(params, ...)`へ明示的に渡し、学習終了時の埋め込みをそのままVoyagerに渡します。
2. **正例/負例と評価リーク** — 旧分割と旧学習処理がデータ加工の都度ファイルを読み直す構造で、分割の追跡が難しいものでした。改訂版ではtrainのエッジのみでGNNメッセージパッシングし、未知として扱うval/test正例はグラフ構築から除外。負例は元の正例全体（逆向きのエッジも含む）から除外し、相互のエッジは同じ分割に割り当てます。
3. **ID対応** — `.unique()` や `.sort()`、one-hot作成とVoyager内の行番号を別々に扱うとIDのズレが起こり得ます。今回、外部node_id→連番→Voyager item_idの対応を明示しました。
4. **指数的なパス増大** — 旧`helper_test_main_loop.py`に10ホップに至るまでの各hop関数と `top_k ** hops` の行展開がありました。今回は指定回数を1つのループで処理し、beam幅上限で展開を制限。
5. **ハードコードパスと副作用** — 過去のコードには個別環境に依存するパス、グローバルパラメータ、ファイルI/O、巨大なデバッグ出力が散在。関数の入出力を型と引数で明示、I/Oを`data.py`/`artifacts.py`に寄せました。
6. **LLMとKGの混同を回避** — ANNで拾う候補列は「グラフの実エッジでつながった経路」とは限りません。旧READMEのコンセプトは維持しつつ、コンテキスト構築の段階でこの区別を明記しました。

## 意図的な非互換性・未移植

- 旧 `GraphNetwork` / Haiku / Flax / Jraph に代わって、**純JAXの2層GraphSAGE-style weighted mean aggregator** を明示実装。元コードのハイパーパラメータ、エッジ重みの意味、GAT代替を完全再現するものではありません。
- 旧KGの２種類の探索起点（目標性能、新たな懸念事項）や、Excel、Marqo上のベクターストア、質問生成、70Bモデル、Web UI、永続化済み結果は元データとサービスがないため収録しません。
- ノード属性は架空の `topic` + `kind` をone-hotにしたもの。実案件の`importance_src` / `importance_dst`の数学的な意味を代替したわけではありません。
- `train()` は小型例におけるミニマルなAdamを用います。実運用ならOptaxを採用し、機能・テスト・ログを整備することを推奨します。

## 面接での正確な表現

「元のプロトタイプは実データと運用検討を含む大規模な構成でしたが、面接でコードレビューできるようコアを分離したデモです。各工程の変更点は表にしました。実行例は架空データなので、元案件の成果指標を再現したとは主張しません。」
