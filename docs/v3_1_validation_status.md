# v3.1 検証状況

実施項目（作成環境）:

- `src/` および `tests/` のPython構文チェック: 合格。
- `compose.yaml` のYAML解析: 合格。
- `train.depends_on.prepare-data.condition == service_completed_successfully`: 確認。
- `prepare-data` の `./data:/app/data` (RW)、`train` の `./data:/app/data:ro` (RO): 確認。
- `prepare-data` と `train` の `GNN_RAG_DATASET` 設定の対応: 確認。
- Polars/Voyager未使用の選択テスト40件: 合格。

未実施:

- Dockerイメージの実ビルドとコンテナ起動（この作成環境にはDocker Engineがない）。
- 実際のHetionet TSV/SIF.gzの取得と変換、および全パイプライン統合テスト。
- Polarsを必要とする既存テスト全体（この作成環境にPolarsがない）。

Macでの統合検証コマンド:

```bash
docker compose config --quiet
docker compose --profile test run --build --rm test
docker compose up --build -d --force-recreate
docker compose ps -a
docker compose logs --no-color --tail=120 prepare-data train api
curl -fsS http://127.0.0.1:8000/info
curl -fsS http://127.0.0.1:8000/evaluation
python3 scripts/smoke_compose.py --generate
```

検証範囲を超えた性能・臨床的妥当性は保証していません。
