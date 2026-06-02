# Local vLLM Endpoint for PSN

This is the **optional** self-hosted LLM path. PSN works out of the box against the OpenAI API (`OPENAI_API_KEY` in `.env`). Use vLLM if you want to:

- run a local / on-prem model for reproducibility,
- avoid per-call API costs on long runs,
- or reproduce the **Qwen/Qwen3-Coder-Next-FP8** results from the paper.

The verified configuration for the paper's diamond-pickaxe run is **Qwen/Qwen3-Coder-Next-FP8 on 2× L40s** (the 80B FP8 tier below).

## Hardware tiers

Pick the row that matches your hardware. All three serve the same OpenAI-compatible API on port 8000.

| Tier | Model | GPUs | `--tensor-parallel-size` | `--max-model-len` |
|---|---|---|---|---|
| **80B FP8** (paper-verified) | `Qwen/Qwen3-Coder-Next-FP8` | 2× 48 GB (e.g. L40s) | `2` | `32768` |
| 30B FP8 | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | 1× 48 GB | `1` | `32768` |
| 80B BF16 | `Qwen/Qwen3-Coder-Next` | 4× 48 GB | `4` | `32768` |

`--gpu-memory-utilization 0.92` works for all three on L40s; lower it (e.g. `0.85`) if you see OOM on other GPUs.

## Bare `vllm serve` (recommended for a dedicated GPU box)

Install vLLM in its own environment (it has a heavy dependency tree; do **not** install it into the PSN env):

```bash
python -m venv ~/.venvs/vllm && source ~/.venvs/vllm/bin/activate
pip install vllm
```

Then launch the model (80B FP8 example):

```bash
vllm serve Qwen/Qwen3-Coder-Next-FP8 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --host 0.0.0.0 --port 8000
```

The first launch downloads ~80 GB of weights to `~/.cache/huggingface/` (set `HF_HOME` to override).

## Docker Compose (portable, useful for remote / CI)

`docker-compose.yml`:

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ports: ["8000:8000"]
    volumes:
      - huggingface-cache:/root/.cache/huggingface
    command: >
      --model ${MODEL_ID:-Qwen/Qwen3-Coder-Next-FP8}
      --tensor-parallel-size ${TENSOR_PARALLEL:-2}
      --max-model-len ${MAX_MODEL_LEN:-32768}
      --gpu-memory-utilization 0.92
      --trust-remote-code
      --host 0.0.0.0
    deploy:
      resources:
        reservations:
          devices:
            - { driver: nvidia, count: all, capabilities: [gpu] }
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 300s
    restart: unless-stopped
volumes:
  huggingface-cache:
```

Launch:

```bash
docker compose up -d
```

Override the model / TP via env:

```bash
MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 TENSOR_PARALLEL=1 docker compose up -d
```

## Verify the endpoint

```bash
curl http://localhost:8000/v1/models
# should return {"data": [{"id": "Qwen/Qwen3-Coder-Next-FP8", ...}], ...}
```

## Point PSN at it

In your `.env`, set:

```bash
VLLM_API_BASE="http://localhost:8000/v1"
VLLM_API_KEY="EMPTY"                          # only needed if you launched vllm with --api-key
VLLM_MODEL="Qwen/Qwen3-Coder-Next-FP8"        # must match the model you launched
```

When `VLLM_API_BASE` is set, every PSN agent (planner / action / critic / curriculum / optimizer) defaults to this endpoint. Per-agent overrides (e.g. send only the critic to OpenAI) are documented in `.env.example`.

`detect_model_profile` in `skillnet/core/model_profile.py` matches `qwen3-coder` as a substring, so any Qwen3-Coder family name automatically picks up the Qwen3-tuned prompt overlay (`control_primitives_context_qwen3/placeItem.js`) and matched temperature / sanitizer settings.

## Notes on cluster reproducibility

If you serve the model on an HPC cluster via a job scheduler (e.g. SLURM), the `vllm serve` invocation above is the substantive part — only the surrounding job-scheduler boilerplate (module loads, storage paths, GPU reservation) differs by site, and that is environment-specific rather than part of PSN. A bare SLURM script is not shipped here because it would only encode one site's module system and paths.
