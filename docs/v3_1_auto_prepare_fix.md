# v3.1：Hetionet前処理の自動化と学習前提条件の修正

## 再現した障害

```text
✘ Container gnn-rag_v2-train-1  service "train" didn't complete successfully: exit 1
FileNotFoundError: Hetionet CSVs missing. Place raw TSV/SIF files in data/raw/ and run
`docker compose --profile hetionet run --rm prepare-hetionet` first.
```

CSVを生成する `prepare-hetionet` が任意の `hetionet` プロファイル内にあり、
`docker compose up` では通常実行されない一方、`train` の `training_options()` が
CSVの存在だけをチェックしていたことが根本原因です。

## 変更点（追跡可能な実装）

| ファイル | 変更内容 |
| --- | --- |
| `compose.yaml` | `prepare-data` サービスを常時起動し、`train` が完了成功を待ってから開始。データvolumeをRW/ROで共有 |
| `src/gnn_rag/compose_prepare.py` | データセット判定、CSV＋メタデータ検証、キャッシュ利用、公式データ取得、部分グラフ変換 |
| `src/gnn_rag/compose_train.py` | CSVが欠損した場合に `prepare-data` ログ確認を促す診断へ変更 |
| `src/gnn_rag/public_data.py` | `dataset_meta.json` に抽出パラメータ（`requested_max_nodes` と `seed`）を追記 |
| `.env.example` | Hetionetの自動前処理を既定の利用例として明示。`HETIONET_AUTO_DOWNLOAD` 追加 |
| `tests/test_compose_prepare.py` | 合成データのネットワーク抑止、既存CSV再利用、欠損時の自動生成、破損検知、パラメータ変更検知など |

## 振る舞い

- `GNN_RAG_DATASET=synthetic`: `prepare-data` は何もダウンロードせず正常終了。
- `GNN_RAG_DATASET=hetionet` ＋ CSVが有効: `prepare-data` は既存のCSVを再利用して正常終了。
- `GNN_RAG_DATASET=hetionet` ＋ CSVが欠損／破損: `data/raw/` にある公式データを検証。なければ自動取得。`data/hetionet_demo/` に再生成。
- `HETIONET_FOCUS`・`HETIONET_MAX_NODES`・`GNN_RAG_SEED` の変更: 新しい条件で部分グラフを再生成。
- 公開データ取得が失敗した場合: `prepare-data` が失敗し、`train` は開始しない。**合成データへの暗黙のフォールバックは禁止。**
- v3旧版の `dataset_meta.json` には抽出seed・最大ノード数の情報がありません。規定の120ノード上限・seed 42に限り、構造が妥当なら再利用します。それ以外の条件は再生成します。

## 新しいサービス依存関係

```text
prepare-data (one-shot) ── success ──> train (one-shot) ── success ──> api
ollama (healthy) ──> model-pull (one-shot) ──────────── success ──> api
```

`prepare-data` と `train` の `Exited (0)` は正常です。

## Mac上での動作確認

```bash
# ZIPルートで、まず .env を作成（既存のものがあれば保持）
[ -f .env ] || cp .env.example .env
# 明示的にHetionetを選ぶ
grep '^GNN_RAG_DATASET=' .env
# この行は hetionet になっていること

docker compose up --build -d --force-recreate
docker compose ps -a
docker compose logs --no-color --tail=150 prepare-data train
curl -fsS http://127.0.0.1:8000/info
curl -fsS http://127.0.0.1:8000/evaluation
python3 scripts/smoke_compose.py --generate
```

`prepare-data` のログに `using prepared Hetionet cache` または
`ready: ... nodes / ... edges` があり、`train` が `Exited (0)`、
`api` が `Up (healthy)` なら初期化完了です。

ネットワーク障害・Git LFSポインタ・データ形式異常時は、次を確認します。

```bash
docker compose logs --no-color --tail=200 prepare-data
ls -lh data/raw/ data/hetionet_demo/
docker compose --profile test run --build --rm test
```

## 検証範囲と制約

ユニットテストで検証したのは、自動前処理の分岐、キャッシュ利用、破損データの検知、引数変更の挙動などです。
**実際のDocker起動・Hetionetダウンロード・GNN学習・Voyagerベンチマークの統合実行は作成環境で未実施です。**
実データには元データの利用条件があるためZIPに同梱していません。
また、この修正は起動の信頼性を改善するもので、GNNの性能優位性や医学的回答品質を保証するものではありません。
