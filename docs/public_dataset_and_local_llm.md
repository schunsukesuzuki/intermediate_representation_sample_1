> **v3.1更新:** `GNN_RAG_DATASET=hetionet` では通常の `docker compose up --build -d` が前処理を自動実行します。手動の `prepare-hetionet` は任意のデバッグ用です。詳細は [`v3_1_auto_prepare_fix.md`](v3_1_auto_prepare_fix.md) を参照してください。

# 公開KG + 小型ローカルLLMの拡張（面接用）

## 目的・要件

- 以前のGPT-4o呼び出しは、OpenAI API等を使わない **OllamaのローカルHTTP API**（`/api/chat`）に置き換えた。
- 学習と検索は **JAX → Polars → Voyager** のまま。LangChain/Marqo依存なし。
- `grounded.py` がANN候補を**観測済みKGエッジ**に照合し、説明に使う根拠を`[E1]`形式で採番。
- **最初の評価では未観測エッジの予測精度と生成回答の正確さを混同しない**。LLMが `[E1]` を書いても実際の文章が根拠に忠実かは人手確認を要する。

## 最短手順（合成データ、LLM不使用）

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,api]'
python -m gnn_rag.rag --epochs 25 --start-id n00
pytest -q
```

確認するファイル: `outputs/local_rag/grounded_prompt.txt`、`outputs/local_rag/evidence_paths.json`、`outputs/local_rag/train_edges.csv`。

ローカルLLMを起動して回答まで試す:

```sh
ollama pull qwen3:1.7b
# Ollama Appが起動していない場合のみ ollama serve
python -m gnn_rag.rag --epochs 25 --start-id n00 --generate --model qwen3:1.7b \
  --question 'このノードに関係する概念を根拠ID付きで列挙してください。'
```

小型の接続確認: `ollama pull qwen3:0.6b` と `--model qwen3:0.6b`。

## 公開データ Hetionet v1.0

出典: [hetio/hetionet](https://github.com/hetio/hetionet)。ノード・エッジを提供するリポジトリの `hetnet/tsv/` 内：

- `hetionet-v1.0-nodes.tsv`: `id, name, kind` のタブ区切り
- `hetionet-v1.0-edges.sif.gz`: `source, metaedge, target` のタブ区切りgzip

大容量ファイルはGit LFSを利用しているため、**ブラウザから個別ダウンロード**または `git lfs` 対応で取得する。LFS pointer のままでは変換できない。

```sh
python -m gnn_rag.public_data \
  --nodes data/raw/hetionet-v1.0-nodes.tsv \
  --edges data/raw/hetionet-v1.0-edges.sif.gz \
  --focus "Alzheimer's disease" --max-nodes 120 \
  --output data/hetionet_demo

# dataset_meta.json の focus_id を自動で使用。必要なら --start-id で明示的に上書き。
python -m gnn_rag.rag \
  --nodes data/hetionet_demo/nodes.csv --edges data/hetionet_demo/edges.csv \
  --epochs 30 --generate --model qwen3:1.7b \
  --question 'この疾患の周辺にどのような観測済みの関係がありますか？'
```

`data/hetionet_demo/dataset_meta.json` に、選択した疾患の `focus_id` とデータ規模を保存します。

### サンプリングの考え方

1. 疾患名一致のノードを選び、1ホップ近傍からノード種類を分散させつつ抽出。
2. 2ホップ近傍を追加し、サンプル内に本当に存在するエッジだけを残す。
3. 関係名を`relation`として残し、分類用カテゴリは`topic/kind`に保持。
4. 学習・評価分割は従来の「ノード対」単位で行い、**RAGに使う観測済みエッジはtrainに限定**。

データ前処理を簡易化するため、同じ無向ノード対に複数の関係があれば代表1つに縮約します。**関係ごとのlink predictionをするモデルではありません**。本格的な異種グラフに対しては relation-specific encoder/decoder（R-GCN、DistMult、ComplEx 等）や方向別評価へ拡張する必要があります。

学習前後のリンク予測AUCを表示しても、小規模誘導部分グラフの選択・閉世界ネガティブ・train/test分割に強く依存します。RAG回答の事実正確性や医療用途での性能は意味しません。SIF/TSVには元データの個別ライセンスや由来属性が欠けるため、製品/再配布前に[元リポジトリ](https://github.com/hetio/hetionet#license)のソース別条件とJSON版を確認してください。

## 認識しておくべき技術的課題

- **GNNのノード表現と質問文が直接対応しない**：今回は質問からエンティティを自動抽出せず、開始ノードIDを指定する。
- **GraphSAGE-style aggregation と relation**：現行エンコーダはエッジ種類を利用しない。検索結果にはラベルを保持するが学習時のrelation-aware性はない。
- **KG path vs latent path**：ANNの連続ホップは根拠にならない。今回は観測済み隣接で裏付けた経路だけプロンプトへ出力。
- **RAGライブラリは必須ではない**：小規模の設計検証には明示的なPython関数とHTTPクライアントが適切。運用時はRAGフレームワーク、評価、キャッシュ、実行監視を別途比較する。
- **持ち出しリスク**：医療・顧客情報は投入しない。本例は公开/架空のテストデータ専用。

## Source references

- Hetionet: https://github.com/hetio/hetionet
- Hetionet paper: https://doi.org/10.7554/eLife.26726
- Qwen3-1.7B: https://ollama.com/library/qwen3:1.7b
- Qwen3-0.6B: https://ollama.com/library/qwen3:0.6b
- Ollama chat API: https://docs.ollama.com/api/chat

## FastAPI でデモする（任意）

```sh
python -m gnn_rag.rag --epochs 25 --start-id n00
GNN_RAG_ARTIFACTS=outputs/local_rag uvicorn gnn_rag.api:app --host 127.0.0.1 --port 8000
```

`http://127.0.0.1:8000/docs` の `/evidence` はLLMを呼ばずに根拠のみ返し、`/answer` はローカルOllamaで根拠IDのみを選択し、回答文をサーバーで確定生成する（`GNN_RAG_OLLAMA_MODEL` でモデル変更可能）。以前の `/retrieve` は**latent-only検索**のまま残してあり、比較対象として使える。公開APIにする想定ではない。

## テスト/検証の状況

このアーカイブ作成環境での構文検査は実施。JAX部分は前の面接版で個別検証済み。しかし現環境に `polars` と `voyager` がなく、ネットワーク制限により追加インストールできないため、**本拡張の全工程・実Hetionet変換・Ollama回答・pytestフルセットは未実行**。提供したテストをユーザーのMacで実行し、API仕様と出力を確認する必要がある。LLMの重みを同梱していない。


## Docker Composeでの自動取得（修正版）

`docker compose --profile hetionet run --build --rm prepare-hetionet` を使うと、公式HetionetのTSV/SIFを取得・検証してから既存の抽出処理を実行します。大きなSIFはGit LFS配信を解決するURLから取得し、サイズとSHA-256を検証します。`data/raw/` はGit対象外です。個別の元データライセンスについては、配布元を確認してください。
