# v10 — Qwen後段説明（4方式×個別）

## 設計

Graph Explorer の `4方式をQwenで説明しPDF保存` を押すと、v9と同じ4方式で**原回答**を作ります。
その後、それぞれに対して `OllamaClient.explain_result()` を**独立に1回ずつ**呼び出し、
原回答の説明文を `explanation` として付加します。

| 方式 | 原回答 | Qwenに渡す説明用データ | 後段の説明 |
|---|---|---|---|
| plain_qwen | KGなしのQwen自由生成 | 対象・質問・原回答だけ | Qwenで説明、**KG根拠は一切渡さない** |
| bfs_only | BFSで選択したTrain根拠の確定レンダリング | 原回答＋選択済みTrainエッジの根拠文 | Qwenで説明 |
| gnn_only | GNNで選択したTrain根拠の確定レンダリング | 同上 | Qwenで説明 |
| bfs_gnn | Validation校正の順位融合から確定レンダリング | 同上 | Qwenで説明 |

- 中間出力の`answer`、`catalog`、`selected_evidence_ids`は変更しません。
- 後段の説明は元の医療事実を検証しません。追加の医学的事実・因果・治療助言の生成を抑えるsystem promptを使用しますが、**幻覚防止を保証しません**。
- それぞれの説明は他方式の結果を見ないため、手法間の優劣をモデルが採点する設計ではありません。
- 説明ができなかった場合でも他の回答は残り、`explanation_status=error` または `skipped_no_answer` となります。
- plain_qwenに同じLLMを再度通す処理は独立した検証ではなく、あくまで自己説明です。
- この4方式の比較は**定性的な回答比較**です。10k/100kの3方式MRR@10比較とは区別してください。

## PDFとJSON

`POST /compare-answers` 実行後にプロジェクト直下に以下を生成／上書きします。

- `GNN_RAG_four_way_answers.pdf`：4方式それぞれに原回答・根拠・**末尾のQwen説明**
- `GNN_RAG_four_way_answers.json`：元のレスポンス・後段説明・生成ステータス・エラー

フロントエンドは `http://127.0.0.1:5173/` の Graph Explorer です。

```bash
[ -f .env ] || cp .env.example .env
# Hetionetの場合: .env で GNN_RAG_DATASET=hetionet, HETIONET_MAX_NODES=5000

docker compose --profile test run --build --rm test
docker compose up --build -d
# Graph Explorer でノードと質問を選択し、比較ボタンを押す
ls -lh GNN_RAG_four_way_answers.pdf GNN_RAG_four_way_answers.json
open GNN_RAG_four_way_answers.pdf
```

APIから直接利用する場合（UIを通さない場合は8000ポート）：

```bash
curl -fsS --max-time 1800 -X POST http://127.0.0.1:8000/compare-answers \
 -H 'Content-Type: application/json' \
 -d '{"node_id":"Disease::DOID:10652","question":"関連する概念は何ですか？","hops":2,"top_k":3}'
```

**実行時間**：CPUコンテナ上でQwenを最大8回（plain原回答×1、KG根拠ID選択×3、説明×4）呼びます。
終了まで時間がかかる場合があります。通信エラーの扱いは各方式に記録します。

10k/100k学習と別PDFの生成は旧来どおり：

```bash
docker compose --profile study run --build --rm study
ls Hetionet_10k_100k_comparison.pdf
```
