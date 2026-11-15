# v8 React UI / PDF 自動生成 — 実装ノート

## 新規モジュールと接続点

- `frontend/src/App.tsx`: 4タブのReactコンソール。`/api/health`, `/api/info`, `/api/evaluation`, `/api/nodes`, `/api/relations`, `/api/evidence`, `/api/answer`, `/api/predict-triples`, `/api/benchmark-summary`, `/api/benchmark-report` を使用。UIは訓練済みモデルの読み取り・推論のみを行う。
- `frontend/nginx.conf`: `/api/` リバースプロキシ。外部にはホストloopbackのみ公開。Dockerの組込みDNSを利用してAPIコンテナの起動前でもSPAを表示可能。
- `src/gnn_rag/pdf_report.py`: ReportLabで日本語PDFを作成（DockerはIPAフォントをaptから導入）。Chart/Table + seed別詳細。今回の `overview.json` で受け取った値のみ使う。
- `src/gnn_rag/dual_scale_benchmark.py`: Train loss・Train Triple数・実行時間の平均を集計。2規模の学習成功後、PDFを `outputs/benchmark/` へ生成し、ホストrootのファイルへ原子的にコピーする。過去レポートと混同しないよう新規学習開始時には既存PDFを無効化する。
- `src/gnn_rag/api.py`: `GET /benchmark-report` は生成済みPDFだけを返し、未生成ならHTTP 404。APIはPDFをオンデマンドで再計算しない。
- `compose.yaml`: 通常起動の `frontend` サービスと、比較実験用 `study` の**専用**ホストrootマウント `./:/app/project_root`。この書込み可能なrootマウントは研究ジョブ`study`だけに与える。通常のAPIとUIには書込み権限を与えない。

## データ取得とモデル

元データ（Hetionetフル）・JAX学習・BFS/GNN/融合のモデル評価はv7と同じ。`GNN_RAG_DATASET=hetionet`, `HETIONET_MAX_NODES=5000` を推奨。**メディカルKGの根拠はTrainの観測済みエッジのみを表示**する。`predict-triples`が返す未観測候補は画面でも仮説と明記する。自由文LLMからの医学的断定は回答に採用しない。

## テスト条件

- Python：v8のPDF出力に関するオフラインテスト（合成指標）を実施。これらはHetionetを再学習した実測値ではない。
- PDF：2ページ描画・行の重なり/欠損がないことを確認。実験の集計値はレポート生成時に渡された新規結果のみ。
- React：TypeScript/JSX構文チェック実施。ビルドはNode 22+Viteが必要。
- Docker Compose：作成環境ではDocker daemonなし。Macでの実データE2Eの再実行は未検証。
