# v11: 4方式のシーケンス解釈と最終横断比較

## 対象

v10から継承した `plain_qwen`, `bfs_only`, `gnn_only`, `bfs_gnn` の原回答は変更しません。
全方式の最後のQwen段階は、ユーザー指定の順序で次の2テキストを生成します。

1. **`sequence_interpretation`**：取得したノード列・元の方向を保持した観測済みTrainエッジを読み、
   推測される追加関係は必ず**仮説・未検証**と書く。複数ホップの存在は直接の因果関係や
   医学的真実を意味しない。`plain_qwen` にはKGノード列・エッジ列を一切渡さない。
2. **`explanation`**：上記生成文について、回答内容、方式の寄与、根拠の限界を3～6文で解説。

この2フィールドをQwenのJSON形式制約で受け取り、アプリ側で最低限の型検証をします。
**3～6文の厳格な文数検証はしていません。** 冗長・構文不正時は `explanation_status=error`。

### 実際のユーザープロンプト中核

```text
以下はプログラムが生成済みの「1方式だけ」の出力データです。
同じ方式の結果を最後に解説してください。

まず、得られたシーケンスをもとに、ノードとエッジから、
想定される関係も含め説明を試みてください。

その上で、出力の書き直しではなく、上記の生成文について、
回答が何を言っているか・方式がどう結果に関係するか・
根拠の限界を3〜6文の日本語で説明してください。
```

後続の検証・ラベルづけ指示と、JSONでの `context` が追加されます。

## 最後の横断比較

すべての個別説明処理の**後**、`compare_four_explanations` は4方式の原回答、
シーケンス解釈、個別説明、各方式で**選択されたTrainエッジのみ**を一括で受け取り、
共通点、相違、推測された関係、検索方式、根拠の限界を考察します。
`cross_comparison` / `cross_comparison_status` をJSONレスポンスに追加。
処理に失敗した場合は `cross_comparison_error` に記録しPDF生成は継続します。

## 表示・ファイル

- React Graph Explorer：4枚の方式カードで「シーケンス解釈」「3～6文の説明」を別表示。
- 4枚のカードの**後**に「Qwenによる4方式の横断比較」を表示。
- `GNN_RAG_four_way_answers.pdf`：同じ順番で4方式の後に横断比較を収録。
- `GNN_RAG_four_way_answers.json`：元回答・根拠・2段階説明・横断比較・エラー情報。
- `Hetionet_10k_100k_comparison.pdf`：従来のリンクランキング検証を維持。Plain QwenのMRRを捏造しない。

## 実行

```bash
[ -f .env ] || cp .env.example .env
docker compose --profile test run --build --rm test
docker compose up --build -d
# ブラウザ: http://127.0.0.1:5173/ → Graph Explorer
# 「4方式を解釈・横断比較しPDF保存」ボタン
ls -lh GNN_RAG_four_way_answers.pdf GNN_RAG_four_way_answers.json
open GNN_RAG_four_way_answers.pdf
```

免責：登録済みKGエッジ≠医学的因果。Qwenが説明した仮説は学習済みモデルやグラフが
検証した関係ではありません。比較文も定量的な優劣の根拠にはなりません。
