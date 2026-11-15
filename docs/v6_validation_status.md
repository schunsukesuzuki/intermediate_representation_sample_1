# GNN-RAG v6 検証記録（2026-11-14）

## 実験の問い

Hetionet Fullを母集団とした約10,000個の**実際の**関係付きTripleで学習し、同じ分割・候補集合でGNN、BFS、融合を評価したらどうなるか。

**実データ結果：未取得・未測定。** ここに記載する数値は処理機構の検証であり、Hetionetでの検索精度を示さない。

## 仕様変更

1. `hetionet_scale.py` で公式フルSIF.gzをストリーミング読み込み、コンパクトな整数配列へ変換
2. ノード型ごとに複数の出発点を選び、ランダムなフロンティア展開で部分グラフを選択
3. 部分グラフから既存の関係付きTripleを集め、ランダムなスパニングフォレストと追加Tripleで**合計10,000件**を選択
4. ノード名は公式TSVの実名をそのまま保持。クラスタIDを合成しない
5. 旧600ノード上限を10kモードでは解除。`HETIONET_MAX_NODES`でメモリ制限
6. 負例候補をTrain Tripleからインデックス化し、epochごとの再計算を軽減
7. 学習時の正例数を4096/epochに制限。GNNのTrainグラフは全Train Tripleを使用
8. 関係別にValidation/Testクエリを事前にサンプリングし、各128クエリ以内で同一候補集合を比較
9. αはValidationのみで最適化。Testには触れず固定

## 行ったテスト

- `tests/test_v6_hetionet_scale.py`: CSV/SIF.gz読み取り、500個の一意Tripleの決定的抽出、キャッシュ有効性、1万Triple抽出、型付き負例、JAX学習、3手法の候補集合/検証
- JAX/NumPyによる**合成**1000ノード・10000 Triple・2 epoch の訓練・テスト関数の完走。結果は `examples/scale_smoke_10k.json` に記録
- ノード47,031/エッジ2,250,197の人工乱数入力を使った大規模サンプラー負荷検査：3,060ノード／10,000Tripleを選択（当該マシンで約1.3秒）。**人工的に生成したグラフであり、Hetionetの構造分布ではない。**
- 既存の単体テストを含む37件を通過（Polarsを要しないsubset）

## 合成グラフの学習出力（実用性能として解釈禁止）

| 項目 | 結果 |
|---|---:|
| ノード | 1,000 |
| 全Triple | 10,000 |
| Train Triple | 7,016 |
| Validation Triple | 1,492 |
| Test Triple | 1,492 |
| epoch | 2 |
| 機能確認用Test評価クエリ | 8 |
| GNN MRR@10 | 0 |
| BFS MRR@10 | 0 |
| 融合 MRR@10 | 0 |

この人工グラフはノード種別しか与えておらず、学習や検索品質を測定するためではない（単に10k規模でプログラムが正常終了するかを確認）。8クエリでのゼロ値に意味のある統計的解釈はできない。

## 実データでの次の検証

Mac上で `docker compose up --build prepare-data train` を実行し、以下の成果物から測定する。

- `data/hetionet_demo/dataset_meta.json`: 実際のデータ全体・抽出量・関係型分布・ノード種別
- `outputs/local_rag/run_summary.json`: Train/Validation/Test、損失、サンプル数
- `outputs/local_rag/evaluation.json` / `.md`: 正規化した同一候補集合の3手法Test MRR、Hits、α
- `outputs/local_rag/evaluation_per_query.csv`: 個々のクエリの診断

**重要な設計限界**：今回の抽出は観測済み全KGの構造を利用してノードを選んでからTrain/Validation/Testに分割する。結果に構造的な選択バイアスがあり、実験を他の論文のランダム分割スコアと直接比較できない。さらにBFSは既知の訓練エッジのみを使うが、GraphSAGEが関係型をMessage Passingに直接使用していない点は未変更。
