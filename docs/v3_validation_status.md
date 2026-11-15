# v3 検証状況（2026-11-14）

## 元となるv2でユーザーが実行・共有した事実

- Hetionet nodes TSV / edges SIF.gzの取得・検証が正常終了。
- Alzheimer’s disease中心の120ノード・315エッジを生成。
- Docker Composeでtrain / model-pullがExited(0)、ollama / apiがhealthy。
- GET /infoで `observed_edges=221`、`model=qwen3:1.7b`。
- `scripts/smoke_compose.py --generate`で8件の検索パスが返り、Qwen3の自由文生成まで成功。ただし`DrD`/`DlA`誤解釈が発見された。

これは**v2**の稼働結果であり、今回のv3をMac/Dockerで動かした結果ではない。

## v3についてこの作成環境で確認した事実

- Pythonソース、スクリプト、テストの構文コンパイル: 成功。
- Compose YAMLの解析とtest profileの定義確認: 成功。
- JAXのGraphSAGE風エンコーダの小配列forward: 成功。
- 外部KG/Voyagerを必要としない単体テスト（Hetionet downloader、設定、Ollama JSON制約、関係型マップ、評価アルゴリズム）: **28件成功**。

## この環境で確認できなかったこと

- PolarsとVoyagerが利用できず、ネットワークから追加インストールもできないため、全pytestとCSV処理・実Voyagerを使ったv3のエンドツーエンドテストは未実行。
- Docker Engineがないため、v3 Dockerfileビルド、`docker compose up`、Qwen3 1.7Bの実モデルによるJSON Schema選択は未検証。
- Hetionet実データでのv3新規ベンチマークの数値（Hits/Recall/MRR）は**未測定**。数値を作り出していない。
- 自由文を使わない回答方式により、以前の誤解釈文を出力経路から排除する設計だが、全ての医学的誤解を防止できると証明したものではない。

## ユーザーのMacで確認すること

```bash
# 別のv3フォルダで。v2を停止してから。
docker compose up --build -d --force-recreate
docker compose ps -a
curl -fsS http://127.0.0.1:8000/evaluation
python3 scripts/smoke_compose.py --generate
cat outputs/local_rag/evaluation.md

# 再現可能な全単体・結合テスト
docker compose --profile test run --build --rm test
```

不一致・失敗がある場合は、`docker compose logs --tail 150 train api` と、`cat outputs/local_rag/evaluation.md` を確認する。
