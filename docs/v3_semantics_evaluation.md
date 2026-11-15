# GNN-RAG v3: 根拠制約付き回答・保留リンク検索評価

## 目的

前版の `DrD` / `DlA` の誤読、疾病と解剖学部位の混同、文献共起から病態・因果を推論する問題を防ぐ。さらに、GNNとVoyagerを組み合わせた検索が本当に有益かを比較できるようにする。

元の約120ノード・315エッジのHetionet部分グラフ、およびJAX/Polars/Voyager/Qwen3・Docker Compose構成を維持する。元データはこのZIPには含まれない。

## 変更 (1) Hetionet関係ラベルを型付きで定義

- `src/gnn_rag/ontology.py` にHetionet v1.0の**24種すべて**のrelation codeを定義。元の略号は保持。
- `DrD`: `Disease–resembles–Disease`。疾患同士の類似を表す。元のソースではMEDLINE文献抄録の共起から推定された関係であり、病理学的因果の実証ではない。
- `DlA`: `Disease–localizes–Anatomy`。`Anatomy` は疾患ではない。文献抄録の共起から推定された関連であり、症例の解剖学的病変を確定するものではない。
- `DpS`: `Disease–presents–Symptom`。こちらも文献共起であり、全症例における必発症状とは読まない。
- エッジの `source/target` のノード型とメタエッジで期待する型を検証。型不一致・未知コードは、証拠カタログには出さない。`Gr>G` の方向付きラベルもそのまま保持。
- 実際の経路探索は逆向きに辿ることもあるが、回答内の根拠エッジの向きを書き換えない。

根拠となる公式スキーマ: <https://github.com/hetio/hetionet/blob/main/describe/definitions.json>、原著のTable 2: <https://elifesciences.org/articles/26726>。

## 変更 (2) 自由文LLM回答ではなく、検証済みIDの選択

1. `grounded_search()` が、学習サブグラフの実エッジから経路を得る。
2. `evidence_catalog()` が `E1, E2, ...` のID・source/target型・関係コードの意味・注記を確定する。
3. Qwen3はOllama `/api/chat` の `format` JSON Schemaに従って `{"evidence_ids": ["E1"], "insufficient_evidence": false}` を返すよう求められる。LLMはID選択のみ担当する。
4. サーバーは受信したJSONを**再検証**する。存在しないID、重複、不正な構造、型不一致は採用しない。
5. 最終回答の医学関連の文は、確定済みのKGエッジと`ontology.py`から**決定的に生成**する。Qwenの自由文は回答に含めない。

スキーマ制約は安全性を保証しないため、サーバー側検証が必須。実在するエッジ自体が医学的正確性を保証するものでもない。

**回答例（イメージ）**：

> [E1] Alzheimer's disease（Disease）とParkinson's disease（Disease）の間には、Hetionet上で「疾患間の類似（文献抄録の共起から推定）」という登録リンクがあります。文献抄録での共起に基づく統計的類似。病態・因果関係の実証ではない。

### 未回答の扱い

LLMが利用できないIDを出したり、不正なJSONを返した場合は、推測した自由文を採用せず、根拠不足の回答を返す（`selection_status` に理由）。モデルが根拠を選ばない場合も同様。**正しい答えを保証するのでなく、未検証の医学的な記述を表示しない設計**。

## 変更 (3) 保留リンクの検索ベンチマーク

実装: `src/gnn_rag/evaluation.py`

**同一テスト集合・同一候補集合**で以下3手法を比較する。

| 手法 | 候補の並べ方 | 注意 |
|---|---|---|
| `gnn_voyager` | 訓練済みJAX GNNのノード埋め込みをVoyager HNSWで近傍検索 | HNSW近似誤差を含む |
| `graph_bfs` | 訓練グラフだけの最短ホップ距離、同距離なら訓練グラフの次数、ノードIDで決定的にソート | 完全グラフ/BFSではない。隠したリンクは使わない |
| `random` | 同一候補集合のランダムな順列。既定20回の繰り返し | 乱数seedを固定、各試行の平均・標準偏差を保存 |

### リーク防止

- `split_edges()` によって無向ペアの両方向を同じ分割に割り当てる。train:val:test = 約70:15:15。
- 訓練用の特徴量とメッセージパッシングは**trainのエッジだけ**を使う。
- 評価クエリはテスト用リンクの両端ノード。**検索候補はすべてのノードから自身と訓練グラフで既知の隣接ノードを除いた集合**。
- GNN・BFS・ランダムすべて同じ候補集合と同じ正解集合で比較。
- テストのリンクはランク付けの後、評価スコア算出にだけ使う。検証リンクも訓練用グラフに入れない。
- 埋め込み評価は**transductive**。全ノードの型・ID・属性は訓練時から知られている。新規ノードの予測は評価しない。

### 定義（各queryノードについてmacro平均）

- `Hits@K`: 上位Kに保留テストの隣接ノードが1個以上あれば1、なければ0。
- `Recall@K`: 上位Kで発見した保留テスト隣接ノード数 / そのクエリの保留テスト隣接ノード総数。
- `MRR@K`: 最初に見つかった保留テスト隣接ノードの順位の逆数。上位Kにない場合は0。

`K=(1,3,5,10)`を既定とする。**これはKGで保留したリンクの発見精度**であり、質問回答の正確性、医療の妥当性、ANNの全件近傍一致率や速度を評価するものではない。

### 制約と正直な読み方

- サンプルされた部分グラフはAlzheimer's diseaseを中心に選ばれた偏りのあるグラフであり、全Hetionetの代表ではない。
- 1つのsplit seedでの結果は統計的有意差を示さない。複数seed・他のグラフ・信頼区間は次の段階。
- テストラベルが不完全で、未評価だが実在する関係が「非関連」として扱われ得る。
- 現行GNNはrelation-awareではない。`DrD`/`DlA`は回答の解釈にのみ使用し、異種エッジごとの重みを学習していない。
- 近傍検索の性能がランダム・BFSより悪い可能性は当然あり、悪い結果もそのまま提示する。
- **GNN自体の寄与を切り出すには** GNNなしの特徴量ベースライン、同じ埋め込みに対するbrute-force kNN、複数seedの再現、適切な学習済み非GNN埋め込み比較が追加で必要。

## 再実行（既存の `.env` と `data/` を保持）

```bash
# 修正版の src/ tests/ docs/ compose.yaml Dockerfile pyproject.toml scripts/
# と .env.example 等をプロジェクトに反映
# 自分の .env と data/raw/、data/hetionet_demo/ を上書きしない

# 旧v2が同じホスト8000ポートを使う場合は先に停止（volumeは残す）
(cd ../GNN-RAG_v2 && docker compose down)
docker compose up --build -d --force-recreate
# train: Exited(0), model-pull: Exited(0), ollama/api: healthy

docker compose ps -a
curl -fsS http://127.0.0.1:8000/info
curl -fsS http://127.0.0.1:8000/evaluation
python3 scripts/smoke_compose.py --generate

# 生成レポート
cat outputs/local_rag/evaluation.md
```

生成ファイル: `evaluation.json`、`evaluation.md`、`evaluation_per_query.csv`、`evidence_catalog.json`、`grounded_prompt.txt`、`run_summary.json`。

## テスト（Dockerのみで実行）

```bash
docker compose --profile test run --build --rm test
```

テスト用の `Dockerfile.test` を独立して用意したため、APIの実行用イメージには `pytest` を含めない。**これまでのテストの結果は手元で未検証**なので、上記コマンドの結果を必ず確認する。

Dockerがない環境ではPython環境にライブラリを導入して `python -m pytest -q` を実行できる。
