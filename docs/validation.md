# 検証状況（環境依存の注意）

このリポジトリは元KG・元モデル重みが提供されていないため、架空データで独立再構成したものです。

**作成時にこの環境で確認できたこと**：全Pythonファイルの構文コンパイル、純JAXのモデルforward、JAXの勾配伝播、20ノードの模擬配列で3epochの学習、学習前後の埋め込みが異なること。

**未確認のこと**：この環境には`polars`と`voyager`がインストールされておらず、ネットワーク制限で追加できなかったため、Polars CSV入出力・Voyagerのバイナリを含む全体の`pytest`とCLI/HTTPのエンドツーエンド実行は未実施です。Voyagerの`add_item`/`query`のAPI仕様はSpotifyの公式リファレンスで確認しました。手元のMacで下記コマンドを実行してください。

検証のためのコマンド:

```bash
python -m compileall -q src tests
pytest -q
python -m gnn_rag.demo --epochs 5 --output outputs/smoke
```

環境によっては依存ライブラリを `pip install -e '.[test,api]'` で導入する必要があります。**元案件の再現性・精度・予測結果の同一性は検証していません。**

### スタックの検証点

- Polars: 入力CSVの列型とnode_idの並び、ノード特徴量の形状。
- JAX: `grad` が学習パラメータへ伝播し、初期値と学習済み埋め込みが一致しないこと。
- Voyager: ベクトル追加→クエリ→元ID復元、自己ノードの除去。
- API: CLIで生成した成果物の読込、無効IDのエラー処理。

### 製品への流用には別途必要

実データでの再評価、デプロイ対象CPU/OSの依存wheel確認、セキュリティレビュー、監視、学習とANNの更新管理、LLM接続のデータガバナンス。
