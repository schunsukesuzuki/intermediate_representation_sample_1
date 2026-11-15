# v4 検証ステータス

## 今回実際に確認したもの

- Python 3.11 / JAX 0.9.0.1 / NumPy 2.3.5を使用。
- v4アルゴリズム部分と既存のオフライン回帰テスト：**48件成功**。
- PythonソースAST解析：成功。
- Compose YAML解析および`prepare-data → train → api`依存関係：成功。
- 30ノード、120Tripleの**人工的な型付きグラフ**で、JAX学習25epochと厳密検索ランキングの数値計算：成功。
- Voyagerの呼び出し・型別インデックス・フィルタリング：小さな**Fake Index**でAPI互換性をテスト（実バイナリではない）。
- 旧v3.1（`schema_version`未設定）からのデータ再前処理判定：検証成功。
- Hetionet SIFの同一ノードペア・異なる関係を失わない変換：PolarsのCSV書出しをテスト用スタブに置き換えて検証成功。

**新しい型付き人工グラフでの数値結果**は `v4_offline_smoke_result.json` を参照してください。ここでは`gnn_exact`のHits@10が0.65、ランダムが約0.853であり、**GNN優位という結果ではありません**。小さい型別候補集合でRandomが高くなることも示します。Voyager列は厳密検索のテストダブルで、実HNSWの値ではありません。

## この作成環境で未確認のもの

- `polars`・`voyager`が未導入（ネットワークから追加インストール不可）。
- Dockerエンジンがない。
- Hetionet公開TSV/SIFの実ダウンロード・v4変換・Docker統合起動。
- Qwen3 1.7Bによるv4エンドツーエンド生成。
- v3.1と同一の型付き候補集合・データ分割を用いたアブレーション。

上記について成功したとは主張しません。実機Macでは以下を検証してください。

```bash
cp .env.example .env
docker compose --profile test run --build --rm test
docker compose up --build -d --force-recreate
docker compose ps -a
docker compose logs --no-color --tail=120 prepare-data train
cat data/hetionet_demo/dataset_meta.json
cat outputs/local_rag/evaluation.md
python3 scripts/smoke_compose.py --generate
```

モデルの意図するインターフェースは`docs/v4_algorithm_and_validation.md`に記載しています。
