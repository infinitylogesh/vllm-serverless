# VLLM Serverless

A private, scale-to-zero vLLM server on [Modal](https://modal.com) with **~70s cold starts** via vLLM sleep mode + Modal GPU memory snapshots.

The full writeup, profiling tables, and the path from a 460s baseline to 70s cold starts is in [blogpost](blogpost.md).

## What this is

- One `@modal.cls` that runs `vllm serve` behind Modal's web server entrypoint.
- On the first cold start, vLLM is started, warmed up (forcing `torch.compile` + CUDA graph capture), and put to sleep. Modal then snapshots CPU and GPU memory.
- On every subsequent cold start, Modal restores from the snapshot and the class wakes vLLM back up. Engine init, compilation, and CUDA graph capture are skipped.

Result: **6.5x faster cold starts** vs. a vanilla `vllm serve`, with no compromise to steady-state throughput (compile, CUDA graphs, and speculative decoding are all on).

## Prerequisites

- A [Modal](https://modal.com) account with GPU snapshot access (currently an alpha feature — request it from Modal if you do not have it).
- `uv` or `pip` for installing the local Python deps.
- A Hugging Face token if the model is gated.

## Setup

1. Install dependencies:

   ```bash
   uv sync
   # or: pip install -e .
   ```

2. Authenticate with Modal:

   ```bash
   modal setup
   ```

3. Create the two Modal secrets referenced in [config.yaml](config.yaml):

   ```bash
   # API key clients will use to call your vLLM endpoint
   modal secret create vllm-api-key VLLM_API_KEY=<pick-any-strong-string>

   # Hugging Face token (only needed for gated models)
   modal secret create huggingface-secret HF_TOKEN=<your-hf-token>
   ```

## Deploy

```bash
modal deploy service.py
```

Modal will:
1. Build the image (cached on subsequent deploys).
2. Spin up a container on an A100-80GB.
3. Download the model into the `huggingface-cache` volume (one-time, slow).
4. Run vLLM, warmup, sleep, and take the GPU snapshot.

> The snapshot is not finalized on the very first cold start. Expect 3–5 cold invocations before snapshot-restore kicks in and cold starts drop to ~70s.

## Calling the endpoint

Modal exposes the vLLM server on a public URL printed by `modal deploy`. The API is OpenAI-compatible:

```bash
curl https://<your-modal-url>/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Or with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-modal-url>/v1",
    api_key="<VLLM_API_KEY>",
)
resp = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)
```

## Tuning knobs

All in [config.yaml](config.yaml):

| Key | What it does |
| --- | --- |
| `model.name` | Hugging Face model id served by vLLM. |
| `model.serve_name` | The `model` value clients pass in API requests. |
| `model.max_model_len` | Override context length. `null` uses the model default. |
| `model.multi_modal` | Toggle image/video input. Disable to shrink warmup. |
| `model.gpu_memory_utilization` | Fraction of GPU memory vLLM may use. `0.85` leaves headroom for snapshot/restore. |
| `service.gpu` | GPU type (`A100-80GB`, `H100`, `L4`, etc.). |
| `service.n_gpu` | Tensor-parallel size. |
| `service.fast_boot` | `false` keeps `torch.compile` + CUDA graphs on (captured in snapshot). `true` uses `--enforce-eager`. |
| `service.scaledown_window` | Idle minutes before scaling to zero. |
| `service.min_containers` | Set to `1` to avoid cold starts entirely (at always-on cost). |
| `service.max_concurrent_requests` | Per-replica concurrency. Tune for your workload. |

### Env vars worth knowing about

Set in [config.yaml](config.yaml) under `service.env`:

- `VLLM_SERVER_DEV_MODE=1` — required to expose the `/sleep` and `/wake_up` endpoints.
- `TORCHINDUCTOR_COMPILE_THREADS=1` — required for snapshot compatibility (multi-threaded inductor state does not snapshot cleanly).
- `SAFETENSORS_FAST_GPU=1`, `HF_XET_HIGH_PERFORMANCE=1` — faster weight loading.
