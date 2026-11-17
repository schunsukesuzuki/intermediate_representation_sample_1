# GNN-RAG v14 — Mac Metal / React / Four-way Qwen / PDF

**Hybrid architecture for an Apple Silicon Mac Studio:** React, FastAPI, and JAX/GraphSAGE run in Docker, while **only Ollama/Qwen3 runs natively on macOS for Metal GPU inference**. Metal is not available to Linux-based Ollama containers running under Docker Desktop. The 10k/100k JAX training workloads also remain CPU-only when run in Docker.

Main changes since v11:

- Removed the hard dependency on Docker's `ollama`/`model-pull` services from the default Compose configuration. The application now connects to `host.docker.internal:11434`. Docker-based CPU Ollama can still be started explicitly using `compose.cpu.yaml`.
- Instead of making a single request wait for up to nine sequential inference calls, progress is polled through `POST /compare-jobs` → `GET /compare-jobs/{job_id}`. The UI displays the current execution stage. **In-progress jobs are lost if the API container restarts** because job state is stored in memory.
- The timeout for an individual Ollama call is 900 seconds (configurable). The model context is 8,192 tokens, and the generation limits for individual explanations and cross-method comparisons are 650 and 850 tokens, respectively (both configurable). Inputs have also been shortened to reduce the risk of context overflow during the four-method comparison.
- When synthetic data is used, the UI explicitly displays **Synthetic demo**. **It is not a Hetionet Graph Explorer unless the model is retrained with `GNN_RAG_DATASET=hetionet`.** `GET /health` exposes `dataset_matches`, and `GET /info` reports the data source used for training.
- Prompts distinguish observed edges that were retrieved but not selected from edges that do not exist, preventing Qwen from asserting that unselected retrieved links are nonexistent. Generated text still consists of **unverified hypotheses**, and medical accuracy is not guaranteed.
- `GET /ollama-status` checks server connectivity and model availability (**actual GPU usage must be verified on the Mac using `ollama ps`**).
- Four-method answers, sequence interpretation, cross-method comparison, PDF/JSON export, and the separate 10k/100k study PDF are retained.

## 1. Start Ollama natively on macOS

Install [Ollama for macOS](https://ollama.com/download/mac). By default, Ollama listens only on `127.0.0.1`. **Only when connecting from Docker Desktop**, configure it as follows on a trusted network.

```bash
# For CLI startup (quit the Ollama app first if it is already running)
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

If you use the GUI app, set the environment variables below, then completely quit and restart Ollama.

```bash
launchctl setenv OLLAMA_HOST '0.0.0.0:11434'
launchctl setenv OLLAMA_CONTEXT_LENGTH '8192'
# Fully quit Ollama from the Dock/Menu Bar, then restart it
open -a Ollama
```

Listening on `0.0.0.0` may also expose the service to other devices on your LAN. **Ollama does not enable authentication by default.** Use the Mac firewall to block unnecessary LAN access. Do not expose the service over public Wi-Fi or through port forwarding. Consider a restricted reverse-proxy configuration as an alternative.

In another terminal:

```bash
ollama pull qwen3:1.7b
ollama run qwen3:1.7b '日本語で短く回答してください。'
# The prompt above asks the model to respond briefly in Japanese.
ollama ps                       # Verify that the PROCESSOR column actually shows 100% GPU or similar
curl -fsS http://127.0.0.1:11434/api/tags
```

**Run `ollama ps` on the Mac, not `docker compose exec ollama ollama ps` inside Docker.**

## 2. Start the v14 Docker Compose stack

Run these commands in the extracted `GNN-RAG-v14-metal/` directory. If an older version such as v11 is still running, execute `docker compose down` in that version's directory. Simply switching project directories does not stop containers belonging to the previous project.

```bash
[ -f .env ] || cp .env.example .env
# If reusing an older .env, check all of the following and change them as necessary
# GNN_RAG_DATASET=hetionet
# HETIONET_MAX_NODES=5000
# HETIONET_TARGET_TRIPLES=100000
# GNN_RAG_OLLAMA_URL=http://host.docker.internal:11434
# GNN_RAG_OLLAMA_TIMEOUT=900
# GNN_RAG_OLLAMA_NUM_CTX=8192
grep -E '^(GNN_RAG_DATASET|HETIONET_MAX_NODES|HETIONET_TARGET_TRIPLES|GNN_RAG_OLLAMA_URL|GNN_RAG_OLLAMA_TIMEOUT|GNN_RAG_OLLAMA_NUM_CTX)=' .env

docker compose config --quiet
# The absence of ollama and model-pull is expected in the default Metal configuration
docker compose config --services
# Run tests
docker compose --profile test run --build --rm test
# Start the stack (also performs the required Hetionet preprocessing and training)
docker compose up --build -d --force-recreate --remove-orphans
docker compose ps -a
```

> If `.env` still contains the old setting `GNN_RAG_DATASET=synthetic`, the main application will train on synthetic data again. **Even when the 10k/100k study uses Hetionet, Graph Explorer uses a separate trained artifact.** Explicitly update the setting. If the original official dataset is required, enable automatic download with `HETIONET_AUTO_DOWNLOAD=1`, or place the original ZIP archive in `data/archives/`.

UI: http://127.0.0.1:5173/  API: http://127.0.0.1:8000/docs

## 3. Diagnose native GPU connectivity and trained artifacts

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/info
curl -fsS http://127.0.0.1:8000/ollama-status
# Check whether the container can reach Ollama running natively on the Mac
 docker compose exec api python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://host.docker.internal:11434/api/tags',timeout=8)))"
# Verify actual GPU usage on the Mac
ollama ps
```

If `/health` reports `dataset_matches: false`, the environment variables and trained artifacts do not match. Editing `.env` alone does not update existing artifacts. Rerun the training service using `docker compose up --build -d --force-recreate`.

If `/ollama-status` reports `model_present: false`, run `ollama pull qwen3:1.7b` on the Mac. If it reports `reachable: false`, check the native Ollama listener and the Mac firewall.

## 4. Four-method answers, progress tracking, and PDF export

Click **“Interpret and compare all four methods, then save as PDF”** in Graph Explorer to start a backend job. The steps run in this order: original answers → interpretation and explanation of each sequence → final cross-method comparison → PDF export. If Qwen fails during interpretation or comparison even after the original outputs have been obtained, the application **explicitly reports the error and saves partial artifacts**.

Files are saved in the project root:

- `GNN_RAG_four_way_answers.pdf`
- `GNN_RAG_four_way_answers.json`

To check the API directly:

```bash
curl -fsS http://127.0.0.1:8000/ollama-status
# Endpoints used by React: POST /compare-jobs, GET /compare-jobs/{job_id}
```

Jobs are stored in the API process's memory. **Do not run `docker compose down` or restart the API while a job is in progress.** There is no persistent queue that can resume interrupted jobs. Only one comparison job can run at a time. Completed PDFs are saved under a fixed filename on the host and will be overwritten by subsequent runs.

**Context and execution time:** `GNN_RAG_OLLAMA_NUM_CTX` sets the context size to 8,192 for each Ollama request. This does not guarantee that every input will fit; the setting is intended to reduce truncation when lengthy evidence is supplied. If Ollama does not respond, each individual call times out after 900 seconds. CPU fallback can be extremely slow.

## 5. Compare 10k/100k training runs (separate job; Metal not used)

```bash
docker compose --profile study run --build --rm study
ls -lh Hetionet_10k_100k_comparison.pdf
```

This experiment trains JAX models on the CPU. It does **not** directly compare MRR@10 against text generated by Plain Qwen.

## 6. Fallback to Ollama inside Docker (CPU)

Only in environments where Metal is unavailable, use the additional Compose file to start CPU-based Ollama. **This will not provide GPU acceleration on the Mac.**

```bash
docker compose -f compose.yaml -f compose.cpu.yaml up --build -d --remove-orphans
docker compose -f compose.yaml -f compose.cpu.yaml ps -a
```

Before switching between CPU and Metal configurations, run `docker compose down` (include the same `-f` options when stopping the CPU stack). Native macOS Ollama and Docker-based Ollama have separate model-storage locations. Avoid adding `--volumes` under normal circumstances, as it removes the models stored on the Docker side.

## 7. Technical and clinical limitations

- Links in the original Hetionet data do **not** establish causality, diagnoses, or treatment effects. Observed edges exist even when they are not selected. LLM-generated hypotheses are not confirmed knowledge-graph facts.
- Plain Qwen is not given KG sequences and is not included in link-prediction metrics such as MRR@10.
- The dataset name displayed in React comes from `run_summary.json`, not from a separate `study` artifact.
- Avoid exposing Docker or Mac host network services externally.
- Actual Metal availability and potential speed gains from native Ollama must be verified with `ollama ps` on the target Mac. GPU integration testing was not performed in the environment used to build the ZIP archive.

For more detailed technical background, see `docs/v11_cross_method.md` and related documentation. In v14, `compose.yaml` prioritizes native Mac Metal, while the older CPU configuration is separated into an explicit supplemental Compose file.

## 8. Typed intermediate representations and evidence auditing inherited from v13

The v12 interface had an issue where Qwen-generated explanations incorrectly stated that **“no KG sequence was provided”** even when actual training edges had been retrieved by BFS, GNN, or the hybrid method. There were also cases where Qwen, while generating the cross-method comparison, confused Plain Qwen's unconstrained output with registered Hetionet edges.

Starting in v13, a **non-generative intermediate representation** was added in `src/gnn_rag/structured_paths.py`. Each retrieved path is validated against `EvidenceFact` before the following fields are deterministically stored in JSON:

- `path_id`, `traversal_node_ids`, `steps[].hop`, `steps[].evidence_id`
- `source_id/name/kind`, `relation_code/label`, `provenance`, `target_id/name/kind`
- `traversal_direction` (`forward` / `reverse`), `selected_for_answer`, `observed_train_edge=true`

An edge's registered direction (`source` → `target`) is distinct from the traversal direction. An observed training edge that is not selected is labeled **“Not selected (edge exists)”**. Unknown edges and discontinuous hops are discarded; the system does not fabricate continuous paths. **This intermediate representation is not generated from Qwen's interpretive text: it is validated and constructed programmatically.**

In React and the PDF, the **deterministically established intermediate representation** appears first, followed separately by Qwen's unverified interpretations and explanations. If Qwen claims that no KG sequence was provided despite the existence of a path, the system detects and explicitly flags this as a misinterpretation. Qwen receives the types, relation labels, provenance, selection status, and traversal directions as input.

Before the cross-method comparison, the program deterministically constructs and displays `method_audit`. Plain Qwen has `no_kg_retrieval` and zero selected training edges, while each of the other three methods retains its own selected `observed_train_kg` evidence. The prompt was also revised so that different relation types (for example, `DlA` and `AeG`) are not treated as the same kind of literature co-occurrence. **Qwen's final cross-method comparison is itself unverified generated text, and there is no guarantee that all misinterpretations have been eliminated. Give priority to the deterministic audit results.**

Tests targeted for verification in v14: `PYTHONPATH=src python -m pytest -q tests/test_v14_structured_generation.py tests/test_v13_structured_intermediate.py tests/test_v12_metal_async.py tests/test_v11_cross_comparison.py tests/test_v10_explanations.py`. Native Metal inference and Docker integration should be tested again on the target Mac.

## v14: Safeguards for Qwen JSON output and factual consistency

- Individual explanations must conform to a JSON Schema requiring `sequence_interpretation` and `explanation`. Incomplete or truncated JSON is rejected.
- If parsing fails or validation is violated, the system shortens the input and retries only once. If the retry also fails, it saves `explanation_error` and `explanation_raw_response` (up to 4,096 characters) in the local `GNN_RAG_four_way_answers.json`.
- Prompts for KG-based methods do not include the **“no path”** instruction intended only for Plain Qwen. Python preserves the original retrieved paths; the LLM is restricted to interpretation and hypothesis generation.
- During the final comparison, Python compares each method's selected edges using `(source,relation,target)`. Method-specific `E#` identifiers, which are renumbered independently, are not used as cross-method identifiers.
- Qwen's final generation contains only two JSON fields: a brief hypothesis and its limitations. Python displays the factual evidence list and renders it into the PDF. If output is truncated, it is retained with `cross_comparison_status=error`.
- Diagnostic responses remain in JSON saved to `GNN_RAG_COMPARISON_DIR`. Before sharing those files, check whether they contain sensitive information when necessary.

See `docs/v14_validation.md` for the v13 failure analysis, v14 changes, and reproducible tests.
