> **v3.1更新:** `GNN_RAG_DATASET=hetionet` では通常の `docker compose up --build -d` が前処理を自動実行します。手動の `prepare-hetionet` は任意のデバッグ用です。詳細は [`v3_1_auto_prepare_fix.md`](v3_1_auto_prepare_fix.md) を参照してください。

# Docker Compose: JAX / Polars / Voyager / Ollama / FastAPI

このリポジトリは公開・合成KGで研究用の検索とローカルLLMをつなぐ**技術面接用デモ**です。患者記録や機密データは入力しないでください。

## 前提

- Docker Desktop (Mac / Apple Silicon を含む) または Docker Engine + Compose v2 (`docker compose version`)。
- CPU実行。Docker上のApple GPU／Metalは使用しません。初回はPython/JAXのイメージ構築とQwen3の重み取得が必要です。CPUとメモリには余裕を持たせてください。
- API公開先は `127.0.0.1:8000` のみ。Ollamaはホストへポートを公開しません。

## 合成データ: 最短実行

展開した `GNN-RAG-docker-compose/` ディレクトリで:

```bash
cp .env.example .env
mkdir -p outputs data/raw
# Composeがtrain・Ollama・モデル取得を先に実行してからAPIを起動
# 初回はモデル取得が必要なのでインターネット接続が必要
docker compose up --build -d
# 起動状況・失敗した処理を確認
docker compose ps -a
docker compose logs --tail 100 train model-pull api
```

`api` が起動したら以下を順に確認:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/info
python3 scripts/smoke_compose.py    # /health /info /evidence
python3 scripts/smoke_compose.py --generate  # さらに /answer
curl -fsS http://127.0.0.1:8000/nodes
curl -fsS -X POST http://127.0.0.1:8000/evidence \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"n00", "question":"この概念に関係する観測済みエッジは？", "hops":2, "top_k":3}'
# こちらはQwen3に実際に問い合わせる。CPU上では応答まで時間を要することがある。
curl -fsS --max-time 360 -X POST http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"n00", "question":"この概念に関係する観測済みエッジは？", "hops":2, "top_k":3}'
```

ブラウザの `http://127.0.0.1:8000/docs` からSwagger UIで同じAPIを試せます。

`outputs/local_rag/`（ホスト側）にモデルパラメータ、埋め込み、観測済みエッジ、プロンプト、検索経路、学習メトリクスが保存されます。`outputs/` はGit対象外です。ダウンロードしたQwen3モデルは `ollama_models` というDocker named volumeに永続化されます。

## サービスと起動順序

| サービス | 役割 | 起動条件 |
|---|---|---|
| `train` | JAXでKGの埋め込みを学習し、Polarsで処理して成果物を保存 | 1回完了して終了 |
| `ollama` | ローカルLLM推論サーバー | 継続起動。healthcheckあり |
| `model-pull` | `ollama` が健康になった後にQwen3を取得 | 1回完了して終了 |
| `api` | Voyager ANN + KG実在エッジ確認 + Ollamaチャット | `train` と `model-pull` の両方が正常終了後に起動 |
| `prepare-hetionet` | 任意のHetionet部分グラフ抽出 | `hetionet` profileで手動実行 |

Composeでは `service_completed_successfully` で1回限りのジョブを待機します。1回目にモデル取得が失敗した場合は `docker compose logs model-pull` を確認し、再実行してください。

## Hetionet v1.0へ切り替える（この修正版で改善）

公式の `hetio/hetionet` からのダウンロード、ファイル検証、部分グラフ抽出を**1つのComposeジョブに統合**しました。初回のネットワーク接続が必要です。特にエッジのSIF gzipはGit LFSファイルで、通常の `raw.githubusercontent.com` では実データではなくLFSポインタが返る場合があります。そのため、このジョブは `media.githubusercontent.com` を用いて実ファイルを取得し、公式LFSのサイズとSHA-256で検証します。ダウンロードしたTSV/SIFは `data/raw/` に保持し、次回は検証して再利用します。

```bash
# 前回データがまだ存在しない状態でも、この1コマンドで取得と変換を実施
mkdir -p data/raw outputs
# 前回の壊れたGit LFSポインタ等を手動配置している場合は先に取り除く
# rm data/raw/hetionet-v1.0-edges.sif.gz

docker compose --profile hetionet run --build --rm prepare-hetionet
# data/hetionet_demo/{nodes.csv,edges.csv,dataset_meta.json} ができたことを確認
ls -lh data/raw data/hetionet_demo

# 初回起動前に .env を用意し、 GNN_RAG_DATASET=hetionet へ変更
cp -n .env.example .env
# .env をエディタで開き GNN_RAG_DATASET=hetionet と設定する

docker compose up --build -d --force-recreate train api
# APIが以前の学習成果物を掴んでいた場合も再起動する

docker compose logs --tail 100 train api
curl -fsS http://127.0.0.1:8000/info
```

`GNN_RAG_DATASET=hetionet` を設定してもCSVが見つからない場合、合成データへ黙って切り替えず、エラーになります。`/info` の `start_id` を問い合わせの `node_id` に指定します（`n00` は合成データ専用です）。

インターネット接続がない場合は、事前に公式配布の `hetionet-v1.0-nodes.tsv` と `hetionet-v1.0-edges.sif.gz` の**実データ**を `data/raw/` へ配置してください。`docker compose --profile hetionet run --rm prepare-hetionet` で検証・変換できます。ファイルが正しければネットワーク接続は不要です。

データの出所は <https://github.com/hetio/hetionet> です。元データを再配布しておらず、TSV/SIFには各エッジのライセンス属性が含まれない点には注意してください。

### 起動失敗時の確認

- `FileNotFoundError: /app/data/raw/...`: 今回の修正版を使って `prepare-hetionet` を再実行します。Composeのvolumesはホストの `./data:/app/data` です。
- `Git LFS pointer`: `data/raw/hetionet-v1.0-edges.sif.gz` のポインタ（小さなテキストファイル）を削除して再実行してください。
- `HTTP 403 / timeout`: GitHub接続を確認するか、公式リポジトリのファイルを手動取得してください。取得できなければ未完成ファイルは残しません。
- `SHA-256 mismatch`: 異なるリリースや破損ファイルを検出しています。公式v1.0のファイルを使用してください。

## モデルを変更する

`.env` の `GNN_RAG_MODEL=qwen3:0.6b` などに変更し、再起動します。

```bash
docker compose up -d --force-recreate model-pull api
```

モデル取得とAPIモデル名は同じ `.env` を参照します。OllamaのモデルはDocker volume上に保存され、`docker compose down` では消えません。削除したい場合だけ `docker compose down -v` を使ってください（重みも消えます）。

## よくある確認事項

- `docker compose` がない: Docker DesktopとCompose v2を導入。
- `api` が起動しない: `docker compose ps -a`、`docker compose logs train model-pull api` を確認。学習または取得が成功するまでAPIは起動しません。
- `port is already allocated`: `.env` の `GNN_RAG_PORT=8001` に変更。
- Apple Siliconで遅い: LinuxコンテナのCPU推論です。`qwen3:0.6b` に落とし、Docker Desktopのメモリ割当を見直してください。
- `/answer` がタイムアウト: 先に `/evidence` と `/health` を確認し、CPU推論のタイムアウトを `.env` の `GNN_RAG_OLLAMA_TIMEOUT` で延長。curlの `--max-time` も適宜調整。
- 実行途中の失敗が出る: `docker compose logs --tail 200 train`、`docker compose logs --tail 200 ollama model-pull`。
- 本番用ではありません: 認証・監査・レート制限等は実装していません。外部へのポート公開、患者データ投入は不可です。

## テストと停止

```bash
# pytestはイメージにオプション[test]を入れていないため、
# ホスト側で `pip install -e '.[test,api]' && pytest -q` を使います。
docker compose down  # Qwen3重みとoutputsは保存したまま停止
```

Docker Engineがない環境ではComposeの静的構文検査までしか実施できません。Dockerでのbuild/起動/実推論は、実機で上記コマンドを用いて確認してください。


### Hetionet nodes.tsv の誤判定（2026-11-14修正版）

TSVヘッダーを改行コードごとのバイト完全一致で比較する方式を廃止し、列名をCSV/TSVパーサーで確認します。CRLF・LF・UTF-8 BOMを許容し、HTMLやGit LFSポインタは拒否します。古いコンテナのコードを使わないよう `docker compose --profile hetionet run --build --rm prepare-hetionet` で再ビルドしてください。無効キャッシュはダウンロードが成功した場合だけ置換されます。
