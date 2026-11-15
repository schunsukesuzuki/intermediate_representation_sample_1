# GNN-RAG v14 — Mac Metal / React / Four-way Qwen / PDF

**Apple Silicon の Mac Studio 用ハイブリッド構成**：React、FastAPI、JAX/GraphSAGE は Docker に置き、**Ollama/Qwen3 のみ macOS ネイティブで Metal GPU 推論**します。Docker Desktop上のLinux OllamaコンテナからMetalは利用できません。10k/100kのJAX学習もDockerではCPUのままです。

v11からの主な変更：

- Dockerの`ollama`/`model-pull`への固定依存をデフォルトComposeから除去し、`host.docker.internal:11434`に接続。Docker版CPU Ollamaは`compose.cpu.yaml`で明示的に起動可能。
- 1回のリクエストで最大9回の逐次推論を待つ代わりに、`POST /compare-jobs`→`GET /compare-jobs/{job_id}`で進捗をポーリング。UIは実行段階を表示します。**APIコンテナが再起動すると進行中ジョブは失われます**（メモリ内ジョブ管理）。
- Ollama単発のタイムアウトは900秒（設定可能）、モデルコンテキストは8,192、個別説明と横断比較の生成上限はそれぞれ650/850 tokens（変更可能）。入力も短縮し、4方式の横断比較がコンテキストを溢れにくく調整。
- 合成データならUIに明確に **Synthetic demo** と表示。`GNN_RAG_DATASET=hetionet`で再学習しない限り、**HetionetのGraph Explorerではありません**。`GET /health`に`dataset_matches`、`GET /info`に学習済みデータソースを表示。
- Qwenが「検索されたが採用されなかったリンクは存在しない」と断定しないよう、プロンプトで未選択の観測エッジと存在しないエッジを区別します。生成文は依然未検証の仮説であり、医学的な正確性は保証しません。
- `GET /ollama-status`でサーバー接続とモデル有無を検査（**GPU稼働はMac側で`ollama ps`による確認が必要**）。
- 4方式の回答・シーケンス解釈・横断比較・PDF/JSON保存と10k/100k別ジョブのPDFは維持。

## 1. macOSでネイティブOllamaを起動

[OllamaのmacOS版](https://ollama.com/download/mac) を導入してください。標準設定は127.0.0.1のみで待ち受けるため、**Docker Desktopから接続する場合のみ**信頼できるネットワーク上で次のように設定します。

```bash
# CLIとして起動する場合（Ollamaアプリがすでに起動していれば先に終了する）
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

GUIアプリ版で動かす場合、次の設定後にOllamaアプリを終了・再起動します。

```bash
launchctl setenv OLLAMA_HOST '0.0.0.0:11434'
launchctl setenv OLLAMA_CONTEXT_LENGTH '8192'
# Dock/Menu barなどからOllamaを完全終了→再起動
open -a Ollama
```

`0.0.0.0`での待受はLAN上にも接続が開く場合があります。**Ollamaは標準で認証なし**です。Macファイアウォールで不要なLANアクセスを遮断し、公開Wi-Fiやポート転送には公開しないでください。代替として限定したリバースプロキシ等の構成も検討してください。

別ターミナルで：

```bash
ollama pull qwen3:1.7b
ollama run qwen3:1.7b '日本語で短く回答してください。'
ollama ps                       # 実際のPROCESSOR欄に100% GPU等が出ることを確認
curl -fsS http://127.0.0.1:11434/api/tags
```

**Macで`ollama ps`を実行してください。Dockerの`docker compose exec ollama ollama ps`ではありません。**

## 2. v14のDocker Composeを起動

このZIPを解凍した`GNN-RAG-v14-metal/`で実行します。v11等をまだ動かしている場合、旧ディレクトリで`docker compose down`を実行してください。プロジェクトを切り替えるだけで古いプロジェクトのコンテナは停止しません。

```bash
[ -f .env ] || cp .env.example .env
# 以前の.envを再利用する場合は下記すべてを確認し、必要なら変更
# GNN_RAG_DATASET=hetionet
# HETIONET_MAX_NODES=5000
# HETIONET_TARGET_TRIPLES=100000
# GNN_RAG_OLLAMA_URL=http://host.docker.internal:11434
# GNN_RAG_OLLAMA_TIMEOUT=900
# GNN_RAG_OLLAMA_NUM_CTX=8192
grep -E '^(GNN_RAG_DATASET|HETIONET_MAX_NODES|HETIONET_TARGET_TRIPLES|GNN_RAG_OLLAMA_URL|GNN_RAG_OLLAMA_TIMEOUT|GNN_RAG_OLLAMA_NUM_CTX)=' .env

docker compose config --quiet
# ここにollamaとmodel-pullがないのがMetalデフォルト構成の正しい状態
docker compose config --services
# テスト
docker compose --profile test run --build --rm test
# 起動（Hetionetの必要な前処理・学習も実行）
docker compose up --build -d --force-recreate --remove-orphans
docker compose ps -a
```

> `.env` が以前の `GNN_RAG_DATASET=synthetic` のままの場合、通常アプリも合成データで学習し直します。**10k/100kのstudy結果がHetionetでもGraph Explorerは別の学習成果物です。** 明示的に変更してください。公式原本が必要なら自動取得（`HETIONET_AUTO_DOWNLOAD=1`）、または`data/archives/`に原本ZIPを置いてください。

UI: http://127.0.0.1:5173/　 API: http://127.0.0.1:8000/docs

## 3. ネイティブGPU接続と学習成果物の診断

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/info
curl -fsS http://127.0.0.1:8000/ollama-status
# コンテナ内からMacネイティブOllamaに到達できるか
 docker compose exec api python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://host.docker.internal:11434/api/tags',timeout=8)))"
# Mac側で実際のGPU稼働確認
ollama ps
```

`/health` が`dataset_matches: false`なら環境変数と学習成果物が不一致です。`.env`を変更しただけでは既存成果物は変わりません。`docker compose up --build -d --force-recreate`で学習サービスを再実行してください。

`/ollama-status` が`model_present: false`ならMacで `ollama pull qwen3:1.7b` を実行してください。`reachable: false`ならMac Ollamaの待ち受け・ファイアウォールを確認してください。

## 4. 4方式の回答・進捗とPDF

Graph Explorerで「4方式を解釈・横断比較しPDF保存」を押すと、バックエンドジョブが進行します。原回答→各シーケンス解釈・説明→最後の横断比較→PDF保存の順です。元の出力が取得できても、Qwenの解釈や横断比較に失敗する場合は**エラーを明記して部分成果物を保存**します。

保存先はプロジェクト直下：

- `GNN_RAG_four_way_answers.pdf`
- `GNN_RAG_four_way_answers.json`

APIから直接確認する場合：

```bash
curl -fsS http://127.0.0.1:8000/ollama-status
# Reactが利用するAPI: POST /compare-jobs, GET /compare-jobs/{job_id}
```

ジョブはAPIプロセスのメモリに保存します。**実行途中に`docker compose down`やAPIの再起動をしないでください**。進行中ジョブを再開する永続キューはありません。同時比較は1件に制限しています。完了PDFはホスト上の固定名で保存されるため、再実行すると上書きされます。

**コンテキストと実行時間**：`GNN_RAG_OLLAMA_NUM_CTX`はOllamaリクエストごとに8,192を指定します。これはモデルが必ず収まることを保証するものではなく、長い根拠が入力される際の切断を抑える設定です。Ollamaが応答しない場合は各呼び出し900秒でエラーになります。CPUフォールバックでは非常に遅くなることがあります。

## 5. 10k/100k学習比較（独立したジョブ、Metal対象外）

```bash
docker compose --profile study run --build --rm study
ls -lh Hetionet_10k_100k_comparison.pdf
```

これはJAXのCPU学習であり、Plain Qwenの生成文とMRR@10を直接比較する実験ではありません。

## 6. Docker内Ollama（CPU）へのフォールバック

Metalが使えない環境でのみ、追加Composeファイルを使ってCPU Ollamaを起動できます。Mac上では**GPUにはなりません**。

```bash
docker compose -f compose.yaml -f compose.cpu.yaml up --build -d --remove-orphans
docker compose -f compose.yaml -f compose.cpu.yaml ps -a
```

CPU/Metalの切り替え時は先に`docker compose down`（CPUスタックなら同じ`-f`指定でdown）してください。モデル保存先はMacネイティブとDocker内で別々です。`--volumes`を付けるとDocker側の保存モデルが消えるので通常は避けてください。

## 7. 技術的・臨床的な制限

- 元のHetionetリンクは因果や診断・治療効果を保証しません。未選択のエッジも、観測済みなら存在しています。LLMの仮説はKGの確定情報ではありません。
- Plain QwenにはKGシーケンスを渡さず、MRR@10等のリンク予測指標には含めません。
- Reactに表示されるデータセット名は `run_summary.json` 由来であり、別の`study`成果物名ではありません。
- DockerとMacホストのネットワークは外部公開を避けて使用してください。
- ネイティブOllamaのMetal使用可否と速度改善は実機で`ollama ps`により確認が必要で、ZIPの作成環境ではGPU統合試験を実施していません。

詳しい技術背景は `docs/v11_cross_method.md` 等を参照してください。v14では`compose.yaml`がMac Metalを優先し、旧CPU構成は明示的な補助ファイルに分離しています。

## 8. v13から継承した型付き中間表現と根拠監査

v12の表示には、BFS/GNN/融合で実際のTrainエッジが取得されているにもかかわらず、Qwenによる説明が「KGシーケンス未提供」と誤記する問題がありました。またPlain Qwenの自由生成を、横断比較のQwenが登録済みHetionetエッジと混同する例がありました。

v13以降、`src/gnn_rag/structured_paths.py` に**非生成的な中間表現**を追加しました。検索で得た各パスを、`EvidenceFact` から検証してから以下をJSONに確定保存します：

- `path_id`, `traversal_node_ids`, `steps[].hop`, `steps[].evidence_id`
- `source_id/name/kind`, `relation_code/label`, `provenance`, `target_id/name/kind`
- `traversal_direction` (`forward` / `reverse`)、`selected_for_answer`、`observed_train_edge=true`

エッジの登録向き（source→target）と探索方向は別物です。未選択のTrainエッジも存在している場合は「未採用（エッジは存在）」と表示します。未知のエッジや不連続なホップは破棄し、連続パスを捏造しません。**この中間表現はQwenの解釈文から生成せず、プログラムが検証・作成します**。

ReactとPDFではまず「確定的な中間表現」を表示し、その下にQwenの未検証解釈・説明を別に表示します。Qwenがパスがあるのに「KGシーケンス未提供」と述べた場合、誤解釈と検出して明示します。Qwenへの入力には型、関係ラベル、出典、採否、探索方向を渡します。

横断比較の前に `method_audit` をプログラムが確定作成して表示します。Plain Qwenは`no_kg_retrieval`で選択済みTrainエッジ0件、残り3方式はそれぞれ独立した`observed_train_kg`の選択済み根拠を保持します。異なる関係型（例：DlAとAeG）を同じ文献共起として扱わないようにプロンプトも修正しています。**Qwenの最終横断比較文そのものは未検証の生成文であり、必ずしも誤解釈がなくなるとは保証できません。確定した監査結果を優先して読んでください**。

v14の確認対象テスト：`PYTHONPATH=src python -m pytest -q tests/test_v14_structured_generation.py tests/test_v13_structured_intermediate.py tests/test_v12_metal_async.py tests/test_v11_cross_comparison.py tests/test_v10_explanations.py`。実機のMetal推論とDockerの統合テストは利用するMacで再確認してください。


## v14: Qwen JSON出力・事実照合の保護

- 個別説明はJSON Schemaで`sequence_interpretation`/`explanation`を要求し、不完全・切断されたJSONを受理しない。
- 解析失敗・検証違反時は、入力を短縮して1度だけ再試行。再失敗時は`explanation_error`と`explanation_raw_response`（最大4096文字）をローカルの`GNN_RAG_four_way_answers.json`へ保存。
- KG方式のプロンプトにPlain Qwen専用の「経路なし」指示を混ぜない。検索済み経路の原文はPythonが維持し、LLMは解釈・仮説のみに限定。
- 最終比較では各方式の選択済みエッジをPythonが`(source,relation,target)`で比較。方法ごとに再付番される`E#`を横断識別子に使用しない。
- Qwenの最終生成は短い仮説と限界の二項目JSONだけ。事実一覧はPythonが表示・PDF化。出力切断時は`cross_comparison_status=error`で保持。
- `GNN_RAG_COMPARISON_DIR`の保存先JSONに診断応答が残るため、必要な場合は機微情報の有無を確認した上で共有すること。

See `docs/v14_validation.md` for v13 failure analysis, v14 changes and reproducible tests.
