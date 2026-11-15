# v12 実装差分・検証記録（2026-11-15）

## 変更点

1. 標準の `compose.yaml` にMacネイティブOllama接続を採用。APIの `GNN_RAG_OLLAMA_URL=http://host.docker.internal:11434`。GPU実行はMacで `ollama ps` を見て確認する。別ファイル `compose.cpu.yaml` ではDocker内CPU Ollamaを使用できる。
2. 4方式比較は非同期API `/compare-jobs` で開始、`/compare-jobs/{job_id}` で進捗ポーリング。従来の同期 `/compare-answers` も互換用に残す。
3. `GNN_RAG_OLLAMA_TIMEOUT=900`、`GNN_RAG_OLLAMA_NUM_CTX=8192`。個別説明650・横断比較850トークンを標準に変更。全体のUIブロッキングを軽減するが、LLMが完走する保証はない。
4. `GET /ollama-status` を追加。Mac APIへの接続とモデル有無を検査する。GPU利用そのものはHTTP接続結果では保証しない。
5. Reactのデータセットを `run_summary.json` に基づき表示。Synthetic と Hetionet の区別が付くようにした。新しいPDFにも dataset 情報を保存。
6. 選択されなかった登録済みKGエッジを「存在しない」と誤断定しないようQwen向けプロンプトを改善。生成した説明の真偽を機械的に検証する機能は追加していない。

## 確認済み

- `PYTHONPATH=src python -m pytest -q tests/test_v12_metal_async.py tests/test_v11_cross_comparison.py tests/test_v10_explanations.py tests/test_llm_container_endpoint.py tests/test_pdf_report_v8.py`: **26 passed**。
- Python `compileall`: エラーなし。
- React `App.tsx`、`api.ts`、`main.tsx`、`vite.config.ts`: TypeScript `transpileModule` によるJSX構文検査を通過。
- `compose.yaml`、`compose.cpu.yaml` はPyYAMLでパース可能。
- サンプル出力を使ったPDFを生成し、レンダリングした日本語表示を確認。

## 実機で要確認

- この作成環境はDockerデーモンがなく、MacネイティブOllama/Metal GPUの統合起動は未検証。
- npmレジストリに接続できないためReactのフル `npm run build` は未実施。構文検査は成功。
- Polars/Voyagerがこの作成環境に無く、全Pythonテストスイート・実データの新規学習は未実施。MacのDocker Composeで全テストを実行する必要がある。
- 4方式の推論および横断比較の医学的事実性・因果的妥当性は未検証。

## 注意

v11 PDFで表示された `看護記録 (n00)` は合成グラフ由来でした。v12でも既存の `.env` が`synthetic`なら、仕様どおり合成データを使用します。Hetionetへ変更するには `GNN_RAG_DATASET=hetionet` および適切なノード上限を設定して再学習してください。`study` のHetionet比較結果はAPI用の学習成果物とは独立です。
