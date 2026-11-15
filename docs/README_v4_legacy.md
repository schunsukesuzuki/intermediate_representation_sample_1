# GNN-RAG v4 — Triple-aware / JAX + Polars + Voyager + Qwen3

**目的:** v3.1の `gnn_voyager Hits@10=0.268 / BFS=0.464` を受けて、モデルを大きくする前に**入力データの意味・学習損失・検索候補の定義**を修正した研究用リポジトリです。**改修後のHetionet精度改善は未実測です。** v3.1とv4の評価課題が異なるため、旧数値との直接比較は禁止です。

## v4で何を変えたか

| 観点 | 旧v3.1 | v4 |
|---|---|---|
| 医療KG | `(source,target)`に統合し関係が失われる | `(source,relation,target)` を全件保持。重複Tripleのみ除去 |
| ノード特徴 | Hetionetでは同じkindを二重one-hot | kind one-hot + 固定hash化した名称3-gram（リンクを使わない） |
| 分割 | 無向ノードペアで分割 | **無向ペア単位の漏洩防止を維持**し、Tripleラベルを保持 |
| 負例 | 全ノードから未接続ペア | 訓練時に同じ `(head,relation)` の妥当なtarget kindから抽出 |
| 目的関数 | 関係なしBCEリンク分類 | GraphSAGE + 関係ベクトル + **型制約付き sampled-softmaxランキング** |
| 検索 | ノード埋め込みの近傍のみ | `(head,relation)`条件付きクエリ。候補型を限定し、厳密検索・Voyagerを同スコアで比較 |
| ベンチマーク | 全ノードの隠した無向ペアを検索 | 隠したTripleを `(head,relation)` ごとにランキング。同一候補集合でExact / Voyager / BFS / Randomを比較 |
| 根拠 | 観測済みエッジのみ | 同じ制約を維持。**既知のKG隣接のみ候補生成し、関係条件付きモデルで再ランキング** |

同じペアに異なる関係がある場合、それらはすべて同じtrain/validation/test分割に入ります。GraphSAGE用のメッセージパッシングは関係数で同一ペアの重みが増えないようペアをユニーク化します。`DrD` 等、同一型の非方向付き関係については訓練と評価のクエリを双方向化します。`Gr>G` は方向付きで保持します。

## Docker Composeで実行する（Mac）

ZIPを展開し、プロジェクトルートで実行します。**旧v2/v3.1の `outputs/` はコピーしない**でください。

```bash
# 旧版の .env があれば引き継ぐ。なければ .env.example を使う
cp .env.example .env
# 必要なら .env の GNN_RAG_DATASET=hetionet を確認
cat .env

# 旧GNN-RAG_v2のAPIが8000番を使用していれば停止（別フォルダの場合）
# (cd ../GNN-RAG_v2 && docker compose down)

docker compose up --build -d --force-recreate
docker compose ps -a
docker compose logs --no-color --tail=100 prepare-data train
```

v4は**旧v3.1の前処理CSVをそのまま再利用しません**。`data/hetionet_demo/dataset_meta.json` の `schema_version` が4でない場合、自動で元の `data/raw/` のTSV/SIF.gzからTriple保持型のCSVを作り直します。`data/raw/` にファイルがなければ公式データを取得します（インターネット接続が必要）。旧版の raw ファイルを `data/raw/` へコピーしておくと再ダウンロードは通常不要です。

```bash
# データ再作成・モデル学習・評価レポートの確認
cat data/hetionet_demo/dataset_meta.json
cat outputs/local_rag/run_summary.json
cat outputs/local_rag/evaluation.md

curl -fsS http://127.0.0.1:8000/info
curl -fsS http://127.0.0.1:8000/evaluation
curl -fsS http://127.0.0.1:8000/relations
python3 scripts/smoke_compose.py --generate
```

### 関係を指定した検索

* **根拠付きKG検索**（observed/train triplesのみ、医学的推論ではない）:

```bash
curl -fsS -X POST http://127.0.0.1:8000/evidence \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"Disease::DOID:10652","relation":"DrD","question":"登録された疾患類似関係を列挙して","hops":1}'
```

* **未観測Triple候補**（予測であり事実・医学的根拠ではない）:

```bash
curl -fsS -X POST http://127.0.0.1:8000/predict-triples \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"Disease::DOID:10652","relation":"DaG","top_k":5}'
```

* **Qwen3は根拠ID選択だけ**を行い、固定の型付きテンプレートから最終回答をレンダリングします。質問からの自動関係特定は未実装のため、明示した `relation` を使ってください。

## 評価の見方

`outputs/local_rag/evaluation.md` には次の4手法のHits@K / Recall@K / MRR@Kを同じ候補条件で出力します。

- `gnn_exact`: 学習済み relation-conditioned query と全候補の厳密スコア。
- `gnn_voyager`: 同じスコアを使い、対象ノード型のVoyager/HNSWで候補を取得。
- `gnn_no_relation`: 同じv4学習済み埋め込みで関係ベクトルだけ使わないクエリ・アブレーション（v3.1を別途再学習した結果ではない）。
- `graph_bfs`: **学習グラフのみ**を利用した最短距離+次数による順位付け。
- `random`: 同じ候補集合をランダムに並べる20試行の平均。

評価で使うTripleは訓練グラフに含まず、無向ペア単位で分割しています。検証用正例はレポート計算時にだけ候補から除外します。精度・一般化能力を主張するには、複数seed・非局所サンプル・より大きなHetionetグラフ等の追加実験が必要です。

**旧数値はv3.1の「全ノード・関係非依存」評価であり、v4の「型制約・関係指定」評価とは母集団が異なります。数値の向上・悪化を直接のモデル改善と解釈しないでください。**

## テスト

```bash
docker compose --profile test run --build --rm test
# ローカルに依存ライブラリがある場合
python -m pip install -e '.[test,api]'
pytest -q
```

研究上の限界: GraphSAGEの近傍集約部分はなお関係非依存（関係別パラメータを持つR-GCNではない）。同一型・非方向関係以外の逆向き検索は別関係として自動推定しない。未観測エッジは真の陰性とは限らない。モデルの候補予測を医療的根拠として使用しない。部分グラフ抽出はAlzheimer's diseaseを中心にするため局所性による選択バイアスがある。小規模ではVoyagerの速度的利点は示せない。

検証結果: [`docs/v4_validation_status.md`](docs/v4_validation_status.md)。

詳細: [`docs/v4_algorithm_and_validation.md`](docs/v4_algorithm_and_validation.md) · [`docs/README_v3_1_legacy.md`](docs/README_v3_1_legacy.md) · [`LICENSE-NOTE.md`](LICENSE-NOTE.md)
