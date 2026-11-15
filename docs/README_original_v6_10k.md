# GNN-RAG v6 — Hetionet Full を母集団にした 10,000 Triple 実験

**実行範囲の注意（2026-11-14）**：コード・データ抽出・JAX学習・評価の機能テストは実行済みですが、**Hetionet本体を使用した1万Tripleの実測スコアは、この配布時点では未測定です**。この実行環境ではGitHub上の元データを取得できず、Polars・Voyager・Dockerの統合起動もできませんでした。**合成グラフのスコアをHetionetの結果として記載しません。** Mac上では自動取得と実測ができます。

## 今回の目的

- 以前の120ノード近傍だけを使う比較から脱却する
- **Hetionet v1.0 全体（約4.7万ノード、225万関係）をサンプリング元とする**
- 初回は約**10,000の一意な関係付きTriple**を抽出し、Train/Validation/Testへ分割する
- GraphSAGE + 関係条件付きランキングの **GNN / BFS / GNN+BFS** を、同一候補集合・同一テストリンクで比較する
- 融合係数αはValidationのみで決定、Testでは固定する
- オリジナルの疾患・遺伝子・薬剤などの**ノード名称**をそのまま使う。合成クラスタ名・隠れたclan識別子は付加しない

**重要な違い：** これは全225万TripleでGNNを学習する設定ではありません。**全データを入力に使い、そこから1万Tripleの構造的サブグラフを抽出する設定**です。抽出処理はData Splitの前に実施します。ランダムIIDサンプルではなく、選択バイアスが存在します。

## 前提

- macOS + Docker Desktop（Apple Siliconは**LinuxコンテナのCPUでJAX実行**）
- ローカルHTTPはFastAPI `127.0.0.1:8000`（本実験ではAPI/Ollamaは任意）
- ある程度のメモリが必要。無理なら後述の規模縮小設定を使う
- 過去のv5.1にある `outputs/` や `data/hetionet_demo/` は**再利用しない**。元データの `data/raw/` のみ再利用可能

## 1. 初回は学習・評価だけ実行（Ollama起動不要）

```bash
# ZIPを展開したGNN-RAG-v6-hetionet-10kの中で実行
cp .env.example .env
mkdir -p data/raw outputs

# 以前ダウンロード済みのファイルがあれば、手元の場所から2ファイルだけコピー可能：
# cp ../GNN-RAG_v2/data/raw/hetionet-v1.0-nodes.tsv data/raw/
# cp ../GNN-RAG_v2/data/raw/hetionet-v1.0-edges.sif.gz data/raw/

# Polars/Voyagerを含む全単体テストはDocker内で：
docker compose --profile test run --build --rm test

# データを自動取得（または正しいrawキャッシュを使用）して、10k抽出→GNN学習→3手法評価
docker compose up --build prepare-data train

# 結果を見る
cat outputs/local_rag/evaluation.md
python3 -m json.tool outputs/local_rag/evaluation.json
cat data/hetionet_demo/dataset_meta.json
```

`train` は正常終了なら `Exited (0)` で正しいです。`prepare-data` も一度だけ実行して終了します。

失敗した場合：

```bash
docker compose ps -a
docker compose logs --no-color --tail=160 prepare-data train
```

### 2. 初期設定値（.env.example）

| 変数 | 既定値 | 意味 |
|---|---:|---|
| `GNN_RAG_DATASET` | `hetionet` | 実データを使う（syntheticにフォールバックしない） |
| `HETIONET_TARGET_TRIPLES` | `10000` | 抽出したTripleの最大目標数（正確に10000を要求） |
| `HETIONET_MAX_NODES` | `5000` | そのために選択してよいノードの上限 |
| `GNN_RAG_EPOCHS` | `25` | 学習epoch数 |
| `GNN_RAG_TRAIN_MAX_POSITIVES` | `4096` | 各epochのランキング学習に使う正例上限。全Trainエッジはメッセージパッシングに使用 |
| `GNN_RAG_EVAL_MAX_QUERIES` | `128` | Validation/Testそれぞれの最大クエリ数 |
| `GNN_RAG_FEATURE_MODE` | `kind_name` | Hetionetの種別と**実在する名前**の文字3-gram。`kind_only` で名前の情報も除去可 |
| `GNN_RAG_SEED` | `42` | データ選択と分割・学習の決定的seed |

`HETIONET_MAX_NODES=5000` で10k Tripleに達しない場合、前処理は正直にエラーを出します。その場合は、**ノード上限を8000〜12000に増やすか、Triple目標を5000に下げる**必要があります。要求した数より少ないデータを使って成功したふりはしません。

**省資源設定：** `.env` を編集し、`HETIONET_TARGET_TRIPLES=3000`、`HETIONET_MAX_NODES=2500`、`GNN_RAG_EVAL_MAX_QUERIES=64`、`GNN_RAG_TRAIN_MAX_POSITIVES=2048` に変更してから再起動。

## 3. 方法の定義

Train/Validation/Testは**同一の無向ノードペアに属するすべての関係付きTripleを1分割に置く**グループ分割です（概ね70/15/15）。学習中のGNNメッセージパッシング/BFSはTrainエッジだけを使用します。Validationで融合重みを選び、TestのMRR@10、Hits@1、Hits@10、Recall@10を報告します。

| 方法 | 関係型の利用 | 順位づけ |
|---|---|---|
| `gnn_only` | 関係ベクトルを使用 | `normalize(z_h + r) · z_t` の厳密全件順位 |
| `bfs_only` | 関係型は候補種別の絞り込みに使用 | Trainグラフ上の距離・次数順 |
| `bfs_gnn` | 両者から利用 | GNN/BFSの順位百分位をαで融合。αはValidationで固定 |

全方法に**同じ関係・ターゲット型の候補集合**を渡します。計測精度の比較では近似HNSWを介さず、GNNスコアは厳密ランキングです。VoyagerはRAG側の検索インデックスとして維持しています（今回の比較のGNNスコアには使用しません）。

**解釈の制約：** 評価クエリは関係型を分散して最大128ずつ選びます。すべての保留Tripleについての全数評価ではありません。エッジの存在しない組を真の負例と見なすことにも限界があります。また、ノード集合を選ぶ段階で全KGの構造を参照するため、データ抽出バイアスは残ります。

## 4. 複数seed評価（上記の単一seedが通った後）

```bash
docker compose --profile benchmark run --build --rm benchmark
cat outputs/multiseed/summary.md
cat outputs/multiseed/summary.json
```

デフォルトは41・42・43の3 seed。**同じ抽出サブグラフ**の上で、split/train/validation αを毎回やり直し、Test平均と標本標準偏差を報告します。これらは独立した医学コホートではなく、記述的な乱数依存性の分析です。

## 5. RAG/LLMのデモも起動したい場合

```bash
docker compose up --build -d
python3 scripts/smoke_compose.py --generate
```

Qwen3の取得には時間とディスク容量が必要です。学習・検索性能評価だけならOllamaは不要です。

## 6. 実施済み検証と未実施の検証

**実施済み**

- 合成の入力TSV/SIF.gzから再現性のある型付きTripleの抽出
- 1万Tripleを厳密な上限で抽出できるストレステスト（ノード名やclan特徴は使用していない）
- 10,000 Triple、1,000ノードの人工グラフでJAX学習2 epoch + 3手法の評価処理を完走
- 回帰・機能テスト（`tests/test_v6_hetionet_scale.py` ほか）
- Compose YAML/Pythonの静的解析

**未実施**

- **Hetionet Fullの原本からの実サンプル抽出とそのMRR/Hitsの測定**（本環境のネットワーク制限）
- Polars・Voyager・Dockerを含む全工程の統合起動（本環境にパッケージ・Dockerなし）
- 実データでの学習規模ごとのLearning Curve

検証詳細：`docs/v6_validation_status.md`。論文・データ権利等の注意：`LICENSE-NOTE.md`。

公式出典: https://github.com/hetio/hetionet — 原本TSV/SIF.gzには個別のライセンス・出典属性が含まれない点に注意してください。
