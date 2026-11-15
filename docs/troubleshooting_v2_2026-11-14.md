# GNN-RAG v2 — エラーログ・修正履歴・動作確認記録

- **記録日**: 2026-11-14
- **実行環境**: macOS / zsh / Docker Compose（ユーザー側のプロジェクト名は `GNN-RAG_v2`）
- **目的**: 技術面接で説明できる GNN-RAG デモを、公開知識グラフ **Hetionet v1.0** とローカル LLM **Qwen3 1.7B** で実行する
- **主要技術**: JAX、Polars、Voyager（ANN / HNSW）、FastAPI、Ollama、Docker Compose
- **最終確認**: Hetionetデータ取得・変換、Composeサービス正常終了／稼働、APIレスポンス、根拠パス抽出、ローカルLLMの回答生成まで、ユーザーの実行ログで確認済み

> **記録の扱い**: エラーメッセージ、コマンド、サービス状態、JSON値は会話で共有されたログを基に記載した。修正箇所については、この作業セッション内にある各ZIPのソースも照合した。原因の一部は実測ではなく「当時の実装に存在した欠陥／もっともらしい説明」であり、該当箇所に明示した。性能・医学的正確性の検証は含まれない。

## 1. 経緯と配布版の対応

| 順番 | 配布ZIP | 変更点・位置付け |
|---|---|---|
| 1 | `GNN-RAG-interview-refactored.zip` | 元の `GNN-RAG-main(1).zip` を基に、データ処理・モデル・学習・検索・コンテキスト生成を分割。学習済みパラメータを推論／埋め込みに反映する設計に修正。合成グラフとテストを同梱。 |
| 2 | `GNN-RAG-public-local-LLM.zip` | Hetionet部分グラフ変換 (`public_data.py`)、実在エッジの検証 (`grounded.py`)、Ollama接続 (`llm.py`)、検索と生成の統合 (`rag.py`) を追加。GPT-4oの代替に `qwen3:1.7b` を採用。 |
| 3 | `GNN-RAG-docker-compose-ready.zip` | `train`、`ollama`、`model-pull`、`api` をComposeで実行。学習成果物をホスト側 `outputs/` に保存。`scripts/smoke_compose.py` を追加。 |
| 4 | `GNN-RAG-docker-compose-hetionet-fix.zip` | **最初の実行エラーを受けて**、公式Hetionetファイルの自動取得・検証を追加 (`fetch_hetionet.py`、`prepare_hetionet.py`)。`prepare-hetionet` ジョブからダウンロード→部分グラフ生成を実行できるようにした。 |
| 5 | `GNN-RAG-docker-compose-hetionet-header-fix.zip` | **2番目の実行エラーを受けて**、ノードTSVのヘッダー検証をバイト列完全一致からCSVパーサーによる列名検証へ変更。LF/CRLF/BOMに対応し、診断表示を改善。**この版で取得・変換・起動・生成の成功ログが得られた**。 |

### 1.1 最終版での主要な構成

```text
Hetionet v1.0 (公式TSV / SIF.gz)
  └─ prepare-hetionet: ダウンロード・検証・120ノード部分グラフ抽出
      └─ data/hetionet_demo/{nodes.csv, edges.csv, dataset_meta.json}
          └─ train: Polars → JAX GNN / Link Prediction → 学習済み埋め込み保存
              └─ outputs/local_rag/
                  └─ api: Voyager ANN → 観測済みKGエッジで経路検証
                      └─ ollama: Qwen3 1.7Bによる根拠付き回答
```

- `train` と `model-pull` は**一回実行して正常終了するジョブ**。`ollama` と `api` は継続起動するサービス。
- Compose設定上、APIの外部公開はローカルの `127.0.0.1:8000` に限定。OllamaはCompose内部の `http://ollama:11434` を使用。
- この版は **Mac上のDockerコンテナでCPU実行**。Metal／GPU利用は実装していない。
- 「RAGフレームワーク」は導入せず、検索・根拠検証・LLM接続を個別のPythonモジュールとして実装。

---

## 2. エラー1: Hetionetの元ファイルが見つからない

### 2.1 発生ログ（抜粋）

```text
File "/usr/local/lib/python3.11/site-packages/gnn_rag/public_data.py", line 73,
  in convert_hetionet
    with open(nodes_tsv, "r", encoding="utf-8", newline="") as handle:
FileNotFoundError: [Errno 2] No such file or directory:
  '/app/data/raw/hetionet-v1.0-nodes.tsv'
```

### 2.2 原因

初期のCompose版は `prepare-hetionet` によるCSV変換を提供していたが、元となる公式TSV/SIFファイルを**自動取得しなかった**。コンテナの `/app/data/raw/` に必要なファイルが存在しなかったため、`public_data.py` が `FileNotFoundError` で停止した。

`compose.yaml` ではホスト側 `./data` をコンテナ側 `/app/data` にマウントしているため、ホスト側にファイルがなければコンテナ内からも参照できない。

### 2.3 修正内容（Hetionet自動取得版）

- `src/gnn_rag/fetch_hetionet.py` を追加。ノードTSVとエッジSIF.gzの取得・内容検証・キャッシュ再利用を実装。
- `src/gnn_rag/prepare_hetionet.py` を追加。データの取得と `convert_hetionet(...)` をひとつのジョブに統合。
- `compose.yaml` の `prepare-hetionet` サービスを更新。`hetionet` profileから明示的に実行可能に変更。
- 取得中は `.part` 一時ファイルを使い、検証に成功した場合だけ正式ファイル名に置き換える。
- エッジSIF.gzはGit LFSポインタでなく実データを取得するため、`media.githubusercontent.com` のURLを第一候補にした。gzipヘッダー、列名、サイズ、SHA-256などを検証する。

**再実行コマンド**:

```bash
docker compose --profile hetionet run --build --rm prepare-hetionet
```

---

## 3. エラー2: ノードTSVを取得したが検証で拒否される

### 3.1 発生ログ（主要部分）

```text
[hetionet] downloading hetionet-v1.0-nodes.tsv from
https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv

[hetionet] downloading hetionet-v1.0-nodes.tsv from
https://github.com/hetio/hetionet/raw/refs/heads/main/hetnet/tsv/hetionet-v1.0-nodes.tsv

RuntimeError: Could not obtain hetionet-v1.0-nodes.tsv.

https://raw.githubusercontent.com/...:
/app/data/raw/.hetionet-2251saos.part is not a valid hetionet-v1.0-nodes.tsv file

https://github.com/...:
/app/data/raw/.hetionet-bp16hha5.part is not a valid hetionet-v1.0-nodes.tsv file

Alternatively, download it manually from https://github.com/hetio/hetionet
into data/raw/ and rerun. Git LFS pointer files are not valid inputs.
```

### 3.2 特定した実装上の不備

前版のノードTSV検証は、ヘッダーの先頭バイトを以下の形で判定していた。

```python
magic = b"id\tname\tkind\n"
```

この方式では、正しいTSVでも行末が `\r\n` (CRLF) だったり、BOMが付いていたりすると、期待するバイト列に一致しない。**この欠陥が存在したことはソースから確認できる**。ただし、失敗時に受信ファイルの先頭バイトを記録していなかったため、**当該エラーの直接原因が実際にCRLFだったかどうかは未確定**。

### 3.3 修正内容（ヘッダー修正版）

- `src/gnn_rag/fetch_hetionet.py` の `validate()` で、ノードTSVを `encoding="utf-8-sig"`、`csv.reader(..., delimiter="\t")` で読み、列名が `['id', 'name', 'kind']` かどうかを確認する方式へ変更。
- LF、CRLF、UTF-8 BOMの差異を吸収。
- 取得結果がHTML、Git LFSポインタ、異常に小さいファイルなどである場合に、診断しやすい例外メッセージを返す。
- 不正キャッシュは、再取得と検証に成功した場合だけ置換する。
- `src/gnn_rag/public_data.py` 側のノードTSV読込も `utf-8-sig` に変更。
- `tests/test_fetch_hetionet.py` を更新。改行・BOM・不正データの検証を追加。

**再実行コマンド**:

```bash
docker compose --profile hetionet run --build --rm prepare-hetionet
```

### 3.4 ユーザー実行ログで解決を確認

```text
[hetionet] downloading hetionet-v1.0-nodes.tsv from
https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv
[hetionet] verified: /app/data/raw/hetionet-v1.0-nodes.tsv

[hetionet] downloading hetionet-v1.0-edges.sif.gz from
https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz
[hetionet] verified: /app/data/raw/hetionet-v1.0-edges.sif.gz
```

続いて出力されたメタデータ:

```json
{
  "focus_id": "Disease::DOID:10652",
  "focus_name": "Alzheimer's disease",
  "nodes": 120,
  "edges": 315,
  "output": "/app/data/hetionet_demo",
  "types": [
    "Anatomy", "Biological Process", "Cellular Component", "Compound",
    "Disease", "Gene", "Molecular Function", "Pathway",
    "Pharmacologic Class", "Side Effect", "Symptom"
  ],
  "note": "Biomedical KG research only; SIF has no per-edge licensing metadata."
}
```

**確定した結果**:

- 公式のノードTSV、エッジSIF.gzをともに取得・検証できた。
- Alzheimer's disease を起点に **120ノード・315エッジ・11種類のノード型**を持つ小規模部分グラフを作成した。
- `data/hetionet_demo/nodes.csv`、`edges.csv`、`dataset_meta.json` の生成処理を完了した。
- 全Hetionetを学習したわけではない。元のSIFにはエッジ単位のライセンス属性はない。

---

## 4. 設定時のエラー: zshで `.env*` が展開されない

### 4.1 発生ログ

```text
GNN-RAG_v2 % ls -la .env*
zsh: no matches found: .env*
```

`ls -la` の結果から、このプロジェクトルートには `Dockerfile` と `compose.yaml` があり、`.env` と `.env.example` は確認できなかった（作業端末でのファイル配置）。`GNN-RAG_v2` はユーザーがリネームしたディレクトリ名であり、名前の変更自体は問題ではない。

### 4.2 原因と対応

これはDockerエラーではなく、**zshが一致するファイルのないグロブ `.env*` をエラーにする挙動**。`.env` が必須でない設定もあるが、Hetionetを選択するには明示設定した方が確実。

会話中に案内した `.env` の内容:

```dotenv
GNN_RAG_DATASET=hetionet
GNN_RAG_EPOCHS=25
GNN_RAG_SEED=42
GNN_RAG_MODEL=qwen3:1.7b
GNN_RAG_OLLAMA_TIMEOUT=300
GNN_RAG_PORT=8000
HETIONET_FOCUS=Alzheimer's disease
HETIONET_MAX_NODES=120
```

新規作成する場合（既存 `.env` を上書きしないこと）:

```bash
cat > .env <<'EOF_ENV'
GNN_RAG_DATASET=hetionet
GNN_RAG_EPOCHS=25
GNN_RAG_SEED=42
GNN_RAG_MODEL=qwen3:1.7b
GNN_RAG_OLLAMA_TIMEOUT=300
GNN_RAG_PORT=8000
HETIONET_FOCUS=Alzheimer's disease
HETIONET_MAX_NODES=120
EOF_ENV
```

確認コマンド:

```bash
cat .env
docker compose config --environment
```

**補足**: ZIP自体には `.env.example` が同梱されている。作業フォルダに見当たらなかった理由（展開・コピー・Git管理など）は、会話ログからは確定できない。zshのエラーだけでファイル損失やComposeの不具合を意味するものではない。

---

## 5. Docker Compose起動とサービス状態

### 5.1 実行コマンド

```bash
docker compose up --build -d --force-recreate
docker compose ps -a
```

### 5.2 最初の状態

| サービス | STATUS（最初） | 説明 |
|---|---|---|
| `api` | `Up ... (health: starting)` | プロセスは起動済み。ヘルスチェックは開始直後で未完了。 |
| `ollama` | `Up ... (healthy)` | Ollamaサーバー稼働、ヘルスチェック成功。 |
| `model-pull` | `Exited (0)` | Qwen3モデル取得ジョブは正常終了。 |
| `train` | `Exited (0)` | GNN学習ジョブは正常終了。 |

その後、`api` も `Up ... (healthy)` に変わった。**最終的には4サービスがすべて期待どおりの状態**になった（`train` と `model-pull` は停止しているのが正常）。

> 注意: `Exited (0)` はプロセスが成功終了したことを意味する。学習内容・モデル品質・検索性能の正しさを単独で証明するものではない。

---

## 6. API確認: Hetionet由来データを読めた

### 6.1 実行コマンド

```bash
curl -fsS http://127.0.0.1:8000/info
```

### 6.2 実行結果

```json
{
  "dataset": "user_csv",
  "start_id": "Disease::DOID:10652",
  "observed_edges": 221,
  "model": "qwen3:1.7b"
}
```

### 6.3 値の意味・315エッジとの差

- `dataset: "user_csv"`: `rag.py` の実装は、**合成データ以外のCSV**を一律 `user_csv` と表示する。Hetionetへの読み替えが失敗したことを意味しない。
- `start_id: "Disease::DOID:10652"`: 前処理で指定したAlzheimer's diseaseのノードIDと一致。
- `observed_edges: 221`: `rag.py` は `split_edges(...)` により分割した後、**訓練側の観測エッジだけ**を `train_edges.csv` に保存し、その本数を返す。`split.py` のデフォルト比率は validation = 15%、test = 15%。315本から約70%の221本になった結果と整合する。
- `model: "qwen3:1.7b"`: APIが回答生成先として設定しているモデル名。

**重要**: この件はログの異常ではない。評価用に分割したエッジをLLMへの「観測済み根拠」に流用しないようにしているため。ただし、個別の分割内訳（validation/test各本数）やモデル指標値そのものは、この時点のログでは提示されていない。

---

## 7. エンドツーエンドの成功ログ

### 7.1 実行コマンド

```bash
python3 scripts/smoke_compose.py --generate
```

### 7.2 返された状態

```json
{
  "health": {
    "ok": true,
    "artifact_directory": "/app/outputs/local_rag"
  },
  "info": {
    "dataset": "user_csv",
    "start_id": "Disease::DOID:10652",
    "observed_edges": 221,
    "model": "qwen3:1.7b"
  },
  "evidence_paths": 8
}
```

続けて:

```text
LLM model: qwen3:1.7b
LLM answer:
[E1] Alzheimer's disease と Creutzfeldt-Jakob disease は、DrD（ドリームド）によって関連している。これは、両者とも神経系の疾患であり、病理学的関連性が観測されているため、観測済みの関係である。

[E2] Creutzfeldt-Jakob disease と amyotrophic lateral sclerosis は、DrDによって関連している。これは、両者とも神経系の疾患であり、病理学的関連性が観測されているため、観測済みの関係である。

[E3] Alzheimer's disease と Parkinson's disease は、DrDによって関連している。これは、両者とも神経系の疾患であり、病理学的関連性が観測されているため、観測済みの関係である。

[E4] Parkinson's disease と amyotrophic lateral sclerosis は、DrDによって関連している。これは、両者とも神経系の疾患であり、病理学的関連性が観測されているため、観測済みの関係である。

[E5] Parkinson's disease と forebrain は、DlAによって関連している。これは、両者とも神経系の疾患であり、解剖学的関連性が観測されているため、観測済みの関係である。

[E6] Creutzfeldt-Jakob disease と cerebellar cortex は、DlAによって関連している。これは
```

上記は**ユーザーが貼り付けた範囲の抜粋**。`[E6]` が途中で切れた理由は、モデル生成上限・CLI表示・コピー範囲などを切り分けていないため不明。

### 7.3 このログで確認できたこと

1. `/health` が `ok: true` を返し、学習成果物ディレクトリを参照できた。
2. `/info` でHetionet由来CSVの起点ノードを返した。
3. `/evidence` で **8件の根拠経路**を返した。
4. `/answer` で `qwen3:1.7b` の日本語回答を返した。
5. よって、**公開KGの変換 → 学習ジョブ → 学習済み埋め込み／Voyager検索 → 根拠検証 → ローカルLLM生成**という接続が、ユーザーのDocker Compose環境で少なくとも1回動作した。

> この成功は「APIの呼出経路とLLM応答が実行できた」ことを意味する。検索精度向上、再現実験での統計的優位性、医学的正確性を示すものではない。

---

## 8. 生成された回答の品質に残る問題（未修正）

以下は**実行エラーではなく、E2E成功後に判明した出力品質上の課題**。現行ZIPに対する追加修正はまだ行っていない。

| 観測した出力 | 問題 | 今後の対処案 |
|---|---|---|
| `DrD（ドリームド）` | 略号を不自然な語として解釈している。Hetionetにおける `DrD` の意味は *Disease resembles Disease*（疾患同士の類似関係）。 | 関係コード→意味・端点型の辞書を用意し、根拠プロンプトで明示。 |
| `Parkinson's disease と forebrain は、両者とも神経系の疾患` | `forebrain` は解剖学的部位であり、疾病と誤分類している。`DlA` は *Disease localizes Anatomy*。 | ノード型（Disease, Anatomyなど）をプロンプトに明示。 |
| `病理学的関連性が観測されているため` | KGの関係ラベルだけから機序や病理学的因果を断定している可能性がある。 | 「KGで記録された関係」と「医学的に裏付けられた説明」を区別し、推測を禁止。 |
| `[E6]` が途中で終了 | 出力切断の可能性。原因は未特定。 | APIの生レスポンス、Ollamaの出力トークン数、ターミナルのコピー範囲を個別確認。 |

その他の既知の設計上の限界:

- 現在のGNNは**relation-awareではない**。異種関係コードは検索根拠の表示には残すが、GNN学習目的で関係種類を区別していない。
- ノードID起点の検索であり、質問文からの自動エンティティリンクは未実装。
- 元のHetionet全件ではなく、120ノードの部分グラフ。全件学習の計算量・評価値は未検証。
- `Hits@K`、`MRR`、`Recall@K` などの比較評価と、通常グラフ探索／ランダム検索などのベースライン比較は未実施。
- 医療上の意思決定には使用しない。公開KGのエッジの存在と因果関係・医学的妥当性は別の問題。

---

## 9. 再現手順（正常動作確認済みの流れ）

プロジェクトルートは現在 `GNN-RAG_v2/` とする。

```bash
# 1. 作業ディレクトリを確認
pwd
ls -la compose.yaml Dockerfile

# 2. Hetionetの取得・変換（取得済みなら検証済みキャッシュを再利用）
mkdir -p data/raw outputs
docker compose --profile hetionet run --build --rm prepare-hetionet

# 3. 生成データの存在確認
ls -lh data/raw data/hetionet_demo

# 4. 設定確認（.envを既に作成済みの場合）
grep '^GNN_RAG_DATASET=' .env
# 期待値: GNN_RAG_DATASET=hetionet

# 5. 学習・モデル取得・APIの起動
docker compose up --build -d --force-recreate
docker compose ps -a

# 6. APIとローカルLLM生成の確認
curl -fsS http://127.0.0.1:8000/info
python3 scripts/smoke_compose.py --generate
```

失敗時の基本調査:

```bash
docker compose ps -a
docker compose logs --tail 200 prepare-hetionet
docker compose logs --tail 200 train model-pull ollama api
cat .env
```

※ `prepare-hetionet` は `docker compose run --rm` で一時コンテナとして実行するため、通常の `docker compose logs prepare-hetionet` では**過去の一時コンテナのログが残らない場合がある**。エラーが発生した場合は、実行中のターミナル出力を保存する。

再実行ログを記録したい場合の例:

```bash
mkdir -p logs
docker compose --profile hetionet run --build --rm prepare-hetionet 2>&1 | tee logs/prepare-hetionet.log
docker compose ps -a | tee logs/compose-status.log
python3 scripts/smoke_compose.py --generate 2>&1 | tee logs/smoke-generate.log
```

`logs/` の内容には設定・実行情報が含まれうるため、公開前に確認する。

---

## 10. 技術面接・GitHub共有時の説明ポイント

**実装したこと**:

- JAX/Polars/VoyagerベースのGNN-RAGを、学習とAPIを分離したモジュラー構成へ再設計。
- 元業務データの代わりにHetionet公開データから小規模な部分グラフを生成。
- 外部のGPT-4o APIに依存せず、Compose内のOllama／Qwen3 1.7Bで日本語回答を生成。
- Git LFS・改行コードなどのデータ取得上の問題を修正し、実際のMac上でE2E実行に到達。
- ANN上の近傍をそのまま真のKGエッジとはみなさず、訓練サブグラフに存在するエッジを根拠として提示。

**主張しないこと**:

- この小規模デモでGNNが検索精度を改善したという主張。
- Hetionet全件に対するスケーラビリティ。
- 医学的妥当性・臨床用途での安全性。
- 現行プロンプトが医学的幻覚を防止できているという主張。

**次の改善候補（本記録時点では未実装）**:

1. Hetionetの関係ラベルとノード型をプロンプトに付与し、医学的な過剰推論を抑制。
2. LLM出力切断の発生条件を確認し、生成上限や出力形式を評価。
3. Link PredictionのAUCに加え、Hits@K、MRR、Recall@Kなどと非GNNベースラインを比較。
4. `pytest -q` とDocker上の再実行を継続的に記録し、再現性を担保。

## 11. 参照したリポジトリ内の主要ファイル

- `compose.yaml` — 5つのComposeサービスと依存関係、ボリューム。
- `.env.example` — Compose設定値のテンプレート。
- `src/gnn_rag/fetch_hetionet.py` — ダウンロード、TSV/gzipの検証、Git LFS対応。
- `src/gnn_rag/prepare_hetionet.py` — 公開データ取得と部分グラフ生成ジョブ。
- `src/gnn_rag/public_data.py` — Hetionet部分グラフ抽出・CSV変換。
- `src/gnn_rag/compose_train.py` — Compose用の学習処理。
- `src/gnn_rag/split.py` — 訓練・検証・テストのエッジ分割。
- `src/gnn_rag/train.py` — JAXによるモデル学習とメトリクス。
- `src/gnn_rag/rag.py` — 学習、Voyager、根拠抽出、LLM接続の統合。
- `src/gnn_rag/grounded.py` — 観測済みエッジに基づく根拠経路作成。
- `src/gnn_rag/llm.py` — Ollama `/api/chat` 呼び出し。
- `src/gnn_rag/api.py` — `/health`, `/info`, `/retrieve`, `/evidence`, `/answer` など。
- `scripts/smoke_compose.py` — API／生成のスモークテスト。
- `tests/test_fetch_hetionet.py` — ダウンロード検証回帰テスト。

---

**最終ステータス（2026-11-14）**: **起動・データ取得・RAG回答生成は成功**。残課題は、主としてHetionetの関係意味論の取り扱い、LLM回答品質、検索性能の定量評価である。
