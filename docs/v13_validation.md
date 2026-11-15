# v13 型付き中間表現と監査の検証（2026-11-15）

## 修正理由（v12実行PDFより）

1. BFS/GNN/融合で観測済みエッジが抽出されたにもかかわらず、Qwen説明に「KGシーケンス未提供」と表示された。
2. Qwen単独の自由生成と、選択済みHetionet Trainエッジが、最終的なQwen横断比較で混在した。
3. 疾患―解剖学的部位の共起 (DlA) と解剖学的部位―遺伝子発現 (AeG) の意味が説明上で同一視される場合があった。

## v13の不変条件

- 構造化パスは既知のCatalogに含まれる `(source, relation, target)` 三つ組のみから構成する。
- `path.nodes[i], path.nodes[i+1]` がエッジの両端と一致しない場合はそこで打ち切る。非登録の中間エッジを飛び越えない。
- 登録エッジの向きと探索方向は別に保存する。逆向きの探索を登録方向としない。
- 関係コード、ラベル、出典・定義、ノード型、採用済み/未選択を失わない。
- Plain Qwenへの解釈入力はKGパスやKG登録エッジを含まない。
- Qwen生成文の「シーケンス未提供」は構造化パスの存在を上書きできない。
- 横断比較の独立監査（`method_audit`）はLLM生成ではなくプログラムが作る。

## 出力先

- APIジョブ結果JSON：`results[method].structured_paths` と `structured_path_text`、`method_audit`
- React：4方式のそれぞれに「確定的な中間表現」、横断比較の手前に「確定的な根拠照合」
- PDF：登録根拠、型付きパス、未検証LLM説明、決定的な根拠照合、未検証LLM横断比較の順

## 限界

- Hetionetの登録リンクから医学的な因果関係を導けない。エッジを2ホップ連結しても、新たな直接リンクが登録されるわけではない。
- `structured_paths` はGraph Explorerが取得したTrainパスのみで、未取得・保留・未観測の全KG関係を列挙するものではない。
- Qwen生成の仮説と比較は引き続き未検証。出典混同などを完全に防ぐシステムではない。
- PDFは研究用説明の再現であり医学的助言ではない。

## ZIP作成時に実行した確認

- 対象の関連テスト：31件通過（v10, v11, v12, v13, PDF v8, LLM URL）。
- `frontend/src/App.tsx`, `frontend/src/api.ts`, `frontend/src/main.tsx` を TypeScript compiler の `transpileModule` で構文検査しエラーなし。
- `v13_preview.pdf`（モック回答を用いた非LLMの動作確認）を生成して3ページを画像描画し、型付きパスと根拠監査のセクションを確認。
- 全体pytest実行は、この作成環境にPolarsなどの依存パッケージがないため実施できていません。Docker統合起動・macOS Metal実機での生成結果の正しさもここでは未検証です。
