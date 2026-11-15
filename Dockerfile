# Native linux/amd64 or linux/arm64 CPU build; no CUDA or jax-metal required.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN apt-get update && apt-get install -y --no-install-recommends fonts-ipafont-gothic \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE-NOTE.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install '.[api]'
EXPOSE 8000
CMD ["uvicorn", "gnn_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
