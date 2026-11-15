# GNN-RAG v14 変更点・検証（2026-11-15）

## v13実機PDFから確認できた事象
- BFS・GNN個別説明が `Qwen explanation was not valid JSON` で失敗した。
- 融合方式のモデル説明が、Python側では実在するTrainパスを「KGシーケンス未提供」と誤認した。
- Qwenの最終横断比較がPlain Qwenと観測済みKG根拠を混同し、途中で文章が切れた。なお、原因が最大生成トークンへの到達かどうかはOllamaメタデータがないため確定していない。

## 実装上の変更
1. `llm_json.py`: JSON Schema出力の必須フィールド・不意のフィールド・最大文字数を検証し、不完全なJSONは明示的な例外として扱う。
2. `llm.py`: 2フィールドJSON形式をOllama `/api/chat`の`format`に渡し、温度0・`think=false`で実行。異常時は入力縮約による再試行を**最大1回**。Ollamaが`done_reason=length`等を返した場合は切断として扱う。
3. `llm.py`: KGモードのプロンプトからPlain Qwenの「KGシーケンスなし」指示が混入する問題を解消。P1等の型・採否・方向情報の短いコピーを渡す。完全な中間表現はPython側に保持する。
4. `four_way_explanation.py`: 回復・失敗を明示し、ローカルJSONに最大4096文字の生応答と試行失敗理由を残す。PDFには生応答を印刷しない。
5. `structured_paths.py`: 方法横断の根拠照合を`(source, relation, target)`で計算。`E1`などの局所IDが異なる関係を表す場合でも混同しない。Plain Qwenの登録Trainトリプル件数は0。
6. `llm.py`: Qwenの最終仕事を「生成された回答の解釈仮説」と「根拠の限界」だけに限定し、JSON Schemaの短い回答とする。実際の選択エッジや差分はプログラムが確定し、ReactとPDFで別表示する。
7. `/comparison-json`: ローカル診断JSONを取得できるリンクを追加。研究対象に機微情報を入れる場合は公開・共有に注意。

## 保証しないこと
- Qwenの説明の意味的正確性、医学的因果・臨床的妥当性、BFS/GNNの定量優劣は保証しない。
- JSON Schemaは応答**形式**を検査する仕組みであり、内容の真偽を自動判定する仕組みではない。出典・経路の観測真偽は検証済みの構造化エッジと決定的な根拠監査を参照。
- Mac StudioのMetal GPU、Ollama 0.40.16実機、Docker Compose本番統合を、この作成環境からは実行していない。

## 検証方法
```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_v14_structured_generation.py \
  tests/test_v13_structured_intermediate.py \
  tests/test_v12_metal_async.py \
  tests/test_v11_cross_comparison.py \
  tests/test_v10_explanations.py \
  tests/test_llm_container_endpoint.py \
  tests/test_pdf_report_v8.py
```
モック応答を用いたPDFの生成・画像描画、React TSXの構文検査も実行する。
