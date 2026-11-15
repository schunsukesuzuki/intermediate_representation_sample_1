.PHONY: install demo test api
install:
	python -m pip install -e '.[test,api]'
demo:
	python -m gnn_rag.demo --epochs 60
test:
	pytest -q
api:
	uvicorn gnn_rag.api:app --host 127.0.0.1 --port 8000

.PHONY: local-rag local-rag-llm
local-rag:
	python -m gnn_rag.rag --epochs 25 --start-id n00
local-rag-llm:
	python -m gnn_rag.rag --epochs 25 --start-id n00 --generate --model qwen3:1.7b

.PHONY: study study-quick
study:
	docker compose --profile study run --build --rm study
study-quick:
	docker compose --profile study run --build --rm study \
		python -m gnn_rag.dual_scale_benchmark --sizes 10000 100000 --seeds 42 --epochs 25
