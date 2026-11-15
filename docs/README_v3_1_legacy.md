# GNN-RAG v3.1 | Docker Compose / Hetionet自動前処理 / 検索性能評価

## v3.1: `Hetionet CSVs missing` を解消

**今回の修正点:** 以前のv3では、`GNN_RAG_DATASET=hetionet` としても
`docker compose up` の学習前に `data/hetionet_demo/nodes.csv` と
`edges.csv` を生成する依存関係がありませんでした。
`train` は存在チェックで `FileNotFoundError` を投げ、APIも起動しませんでした。

v3.1では通常の `docker compose up --build -d` で次の順序になります。

```text
prepare-data (synthetic なら即終了 / Hetionet ならCSVを確認・必要時だけ生成)
       └── completed successfully ──> train (JAX・Voyager・評価)
                                         └── completed successfully ──> api (FastAPI)
ollama ── healthy ──> model-pull ── completed successfully ────────────> api
```

- 既存の `data/hetionet_demo/nodes.csv`、`edges.csv`、`dataset_meta.json` が有効なら再利用。**v3で生成した旧メタデータも、デフォルト設定なら再利用可能。**
- CSV欠損・破損・`HETIONET_FOCUS` / `HETIONET_MAX_NODES` / `GNN_RAG_SEED` の変更時は再生成。
- 元の `data/raw/` に正しいTSV/SIF.gzがあれば再ダウンロード不要。未取得なら公式の公開データから自動取得（ネット接続が必要）。
- `GNN_RAG_DATASET=synthetic` では外部ダウンロードせず、元データも不要。
- `./data` は `prepare-data` に書き込み可能、`train` に読み取り専用で渡します。**Hetionet指定から合成データに無言で切り替えません。**

### Macでの起動（新しいフォルダに展開）

```bash
# ZIPを解凍した GNN-RAG-v3.1-auto-prepare/ 内で実行
# v2と同じ親ディレクトリに置いた場合、既存の設定・データを使えます
cp ../GNN-RAG_v2/.env .env
mkdir -p data outputs
cp -R ../GNN-RAG_v2/data/. ./data/  # あれば。なくても新規取得・前処理します

# GNN_RAG_DATASET=hetionet を確認（合成データでよければ synthetic）
grep '^GNN_RAG_DATASET=' .env
# 8000番ポート競合を避けるため旧v2を停止（旧モデルデータを消さない）
(cd ../GNN-RAG_v2 && docker compose down)

docker compose up --build -d --force-recreate
docker compose ps -a
docker compose logs --no-color --tail=120 prepare-data train
curl -fsS http://127.0.0.1:8000/info
curl -fsS http://127.0.0.1:8000/evaluation
python3 scripts/smoke_compose.py --generate
```

旧ディレクトリがない場合は `cp .env.example .env` で設定を作ってください。
`prepare-hetionet` の手動起動は**不要**です。手動の再サンプリング・デバッグ用途として残しています。
`.env` の `HETIONET_AUTO_DOWNLOAD=0` に設定するとネットワーク取得は行いません
（既に検証済みCSVがあるか、正しい元TSV/SIF.gzが必要です）。

### 原因と検証

- **症状:** `service "train" didn't complete successfully: exit 1`
- **Traceback:** `FileNotFoundError: Hetionet CSVs missing. Place raw TSV/SIF files ...`
- **根本原因:** `prepare-hetionet` に `profiles: ["hetionet"]` があり、通常の `up` では起動せず、`train` に必須CSVがなかった。
- **対策:** `prepare-data` への `service_completed_successfully` 依存と、キャッシュ検査・自動前処理を追加。
- **注意:** 実行先MacでのDockerビルド・Hetionet実データ・Voyager統合動作は、この作成環境では未検証。テストは模擬データを利用しています。

詳細は [`docs/v3_1_auto_prepare_fix.md`](docs/v3_1_auto_prepare_fix.md) を参照してください。

---


## v3 アップデート：根拠IDだけをLLMに選ばせる＋保留リンク検索の定量比較

**この版の追加目的は、前回実動作が確認されたHetionet/Docker構成を維持しながら、医学的関係の説明におけるLLMの創作を抑え、GNN検索の性能を検証することです。**

- Hetionetの24種のメタエッジを型付きで定義。`DrD`（疾患間の類似）と`DlA`（疾患–解剖学的部位の関連）を含め、各リンクの文献共起由来などの注意書きを表示。
- Ollama/Qwen3は **JSON Schemaに従って `E1` 等の根拠IDを選択するだけ**。サーバーが厳密にIDを検証し、最終回答の文を観測済みエッジと定義済み関係ラベルから生成する。架空IDは不採用。**医学的事実の完全保証ではない**。
- GNN + Voyager / train-only BFS / ランダム（既定20試行）で保留リンクを比較し、`Hits@K`・`Recall@K`・`MRR@K` を `outputs/local_rag/evaluation.md`・JSON・CSVへ記録。性能向上を前提とせず、測定結果をそのまま確認する。
- `GET /evaluation` を追加。既存の `/info`、`/evidence`、`/answer`、`/retrieve` を維持。`/answer` に `selected_evidence_ids` と `selection_status` を追加。

### Mac上の既存 `GNN-RAG_v2` からv3への移行

**v3.1の起動例は冒頭の手順を推奨します。** 以下は旧v3からの移行コマンドとして残していますが、前処理用ジョブを手動実行する必要はありません。

```bash
# v3フォルダ内で。旧フォルダが同階層にある前提
cp ../GNN-RAG_v2/.env .env
mkdir -p data/raw data/hetionet_demo outputs
cp -R ../GNN-RAG_v2/data/. ./data/

grep '^GNN_RAG_DATASET=' .env   # hetionet と表示されることを確認
# 旧v2と同時起動するとホストの8000番ポートが競合するため、旧コンテナを停止
(cd ../GNN-RAG_v2 && docker compose down)  # -v は付けない（旧モデルvolumeは残す）
docker compose up --build -d --force-recreate
docker compose ps -a
curl -fsS http://127.0.0.1:8000/evaluation
python3 scripts/smoke_compose.py --generate
cat outputs/local_rag/evaluation.md
```

Docker Composeのサービス間通信は維持。ただし、**v3は別のCompose projectとなるため通常は別のOllama volumeが作成され、初回はQwen3を再取得します**。旧v2のモデルvolumeは`docker compose down`だけでは消えません。`ollama` がhealthy、`train`と`model-pull`がExited(0)、`api`がhealthyになってから確認してください。

- **v3変更詳細・ベンチマーク定義:** [`docs/v3_semantics_evaluation.md`](docs/v3_semantics_evaluation.md)
- **再現テスト（Dockerにpytestを準備）:** `docker compose --profile test run --build --rm test`
- **既知の制約:** 片方のseedにおける小規模グラフでの比較は、臨床・質問応答品質の改善を意味しません。GNNはrelation-awareではありません。

**技術面接の設計説明用**に、既存のGNN-RAGを最小構成で再現したリポジトリです。**JAX / Polars / Voyagerを維持**し、実在KGの観測済みエッジを根拠として選び、GPT-4oの代わりにOllama上の小型Qwen3で根拠IDを選択します。回答文は型検証済みKGから決定的に生成します。元の顧客データ・元学習済み重み・実業務の性能結果は含みません。


### 2026-11-14修正：Hetionetのnodes.tsv取得エラー

以前の版では `hetionet-v1.0-nodes.tsv` の先頭を `b"id\tname\tkind\n"` とバイト完全一致で検証していたため、CRLF (`\r\n`) などの差異で実データも `is not a valid ... file` と誤判定される可能性がありました。本版は `csv.reader` による実際の列名検証（LF/CRLFおよびUTF-8 BOM対応）に変更しました。キャッシュが不正でも、正しいファイルを再取得できた場合だけ置き換えます。

```bash
# このZIPを展開したディレクトリで実行。古いイメージを使用しないよう --build を指定
mkdir -p data/raw outputs
docker compose --profile hetionet run --build --rm prepare-hetionet
ls -lh data/raw data/hetionet_demo
```

エラーが続く場合、修正版は `unexpected nodes header`、`contains HTML`、`First bytes` など、失敗原因を明確に表示します。正しいノードTSVはGitHub上で約2.39MBと表示され、先頭列は `id`, `name`, `kind` です。データ本体とLLM重みはZIPに含めていません。

## Docker Composeで起動する

```bash
cp .env.example .env
mkdir -p outputs data/raw
docker compose up --build -d
docker compose ps -a
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/info
python3 scripts/smoke_compose.py --generate  # /health, /info, /evidence, /answer
```

初回はコンテナイメージの構築・学習・Qwen3重み取得を実施します。`api` は学習とモデル取得が終わるまで起動しません。ログは `docker compose logs train model-pull api` で確認してください。Docker Desktop（Apple Silicon含む）は**CPUのみ**で動く想定です。Dockerによる実動作は作成環境で未検証です。

APIサンプル:

```bash
curl -fsS -X POST http://127.0.0.1:8000/evidence \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"n00", "question":"このノードの観測済み関係を説明してください", "hops":2}'

curl -fsS --max-time 360 -X POST http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"n00", "question":"このノードの観測済み関係を説明してください", "hops":2}'
```

- **詳細な起動方法・Hetionetへの切り替え:** [`docs/docker_compose.md`](docs/docker_compose.md)
- **公開データのライセンス・サンプル抽出:** [`docs/public_dataset_and_local_llm.md`](docs/public_dataset_and_local_llm.md)
- **面接での5分説明:** [`docs/interview_walkthrough.md`](docs/interview_walkthrough.md)
- **元コードとの関係と設計変更:** [`docs/legacy_mapping.md`](docs/legacy_mapping.md)
- **評価計画:** [`docs/evaluation_plan.md`](docs/evaluation_plan.md)

## アーキテクチャ

```text
   synthetic KG  または  Hetionet v1.0 (ユーザーが取得し部分抽出)
             │
             ▼
    Polars: 読込・整合性検証・ノードID対応
             │
             ▼
    JAX: GraphSAGE-style GNN / Link Prediction 学習 ── train (1回実行)
             │
             ▼
    学習済みembeddings.npy + 観測済みtrain_edges.csv (./outputs)
             │
             ▼
    FastAPI + Voyager: HNSW近傍探索
             │
             ▼
    学習で観測したKGエッジと照合 (架空パスの根拠化を防止)
             │
             ▼
    エッジID付きプロンプト ──► Ollama / Qwen3 (ローカルHTTP)
             │
             ▼
    /answer (生成文 + 検索根拠)
```

| API | 機能 |
|---|---|
| `GET /health` | 学習成果物の有無 |
| `GET /info` | データ種別・検索開始ノードID・LLMモデル名 |
| `GET /nodes` | 検索可能なノード一覧 |
| `POST /retrieve` | 埋め込み最近傍のみ（KGエッジ未検証の比較対象） |
| `POST /evidence` | 観測済みエッジで検証した検索経路と根拠プロンプト |
| `POST /answer` | Qwen3が根拠IDを選択し、確定ルールに基づき回答を生成 |

## 主な設計判断

- **学習とAPIを分離:** APIリクエストごとにJAX学習を回さない。検索は保存済みの**学習後**埋め込みを使用する。
- **実在エッジで検証:** ANNで見つけた類似ノードをそのままKG関係と見なさず、学習グラフ中の観測済みエッジを根拠として使う。
- **Ollama接続はCompose内部のみ:** ホストへ公開するのは `127.0.0.1:8000` のAPIだけ。モデルはDocker volumeに永続化。
- **外部ライブラリは最小限:** RAGフレームワークを追加せず、検索とLLM接続の責任を個別のモジュールに分離。
- **医学的妥当性・性能評価を主張しない:** 合成データと小規模誘導グラフの指標は患者ケアの有効性を示さない。

**既知の制約:** GNNはrelation-awareでなく、質問文からの自動ノードリンクも未実装です。ノードIDを指定して検索します。合成グラフの評価値は実データ性能を示しません。医療用途や外部公開用の認証・監査はありません。

## Dockerなしで実行

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,api]'
python -m gnn_rag.rag --epochs 25 --start-id n00
pytest -q
```

LLMをホストのOllamaアプリで利用する場合:

```bash
ollama pull qwen3:1.7b
python -m gnn_rag.rag --epochs 25 --start-id n00 --generate --model qwen3:1.7b
```

## 出力とデータ

- 合成データ: 初回から作成できます。`outputs/local_rag/` に重みと根拠ファイルを保存。
- Hetionet: データ本体は同梱しません。`GNN_RAG_DATASET=hetionet` で通常の `docker compose up` を実行すると、`prepare-data` が公式TSV/SIFを必要に応じて取得・検証し、部分グラフを作成します（初回はネット接続が必要）。取得済みデータは `data/raw/` にキャッシュします。
- モデル重み: 同梱せず、初回 `model-pull` が取得。ライセンスや元データの出典も確認してください。

元の実装との**結果の同一性は保証しません**。旧コーディングの修正内容については [`docs/legacy_mapping.md`](docs/legacy_mapping.md) を参照してください。
