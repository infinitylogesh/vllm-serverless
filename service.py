
import modal
from yaml import safe_load
import os

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config.yaml")
with open(config_path, "r") as f:
    config = safe_load(f)

app = modal.App(config["service"]["name"])

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

N_GPU = config["service"]["n_gpu"]
MINUTES = 60  # seconds
VLLM_PORT = config["service"]["port"]
MODEL_NAME = config["model"]["name"]
SERVED_MODEL_NAME = config["model"]["serve_name"]
MODEL_REVISION = config["model"]["revision"]
FAST_BOOT = config["service"]["fast_boot"]
MAX_MODEL_LEN = config["model"]["max_model_len"]
MULTI_MODAL = config["model"]["multi_modal"]
GPU_MEMORY_UTILIZATION = config["model"]["gpu_memory_utilization"]

# ------------------------------------------------------------
# Image
# ------------------------------------------------------------

vllm_image = (
    modal.Image.from_registry(config["service"]["image"], add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        *config["service"]["pip_install"],
    )
    .env(config["service"]["env"])
    .add_local_file(
        os.path.join(local_dir, "config.yaml"),
        remote_path="/root/config.yaml",
        copy=True,
    )
)

volumes = {
    volume["path"]: modal.Volume.from_name(volume["name"], create_if_missing=True)
    for volume in config["service"]["volumes"]
}

# ------------------------------------------------------------
# Helpers (run inside the container)
# ------------------------------------------------------------

with vllm_image.imports():
    import requests


def _wait_ready(proc, timeout=20 * MINUTES):
    import time

    deadline = __import__("time").time() + timeout
    while __import__("time").time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        try:
            requests.get(
                f"http://localhost:{VLLM_PORT}/health", timeout=5
            ).raise_for_status()
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("vLLM did not become ready in time")


def _warmup():
    # Run a few inference requests to force CUDA graph capture and JIT compilation.
    # These artifacts are included in the GPU snapshot so subsequent cold starts skip them.
    import os

    api_key = os.environ.get("VLLM_API_KEY", "")
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 16,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for _ in range(3):
        requests.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=300,
        ).raise_for_status()


def _sleep():
    # Offload model weights from GPU to CPU and empty the KV cache.
    # Makes the snapshot smaller and avoids snapshotting live GPU kernels.
    requests.post(f"http://localhost:{VLLM_PORT}/sleep?level=1", timeout=60).raise_for_status()


def _wake_up():
    # Reload model weights back to GPU after snapshot restore.
    requests.post(f"http://localhost:{VLLM_PORT}/wake_up", timeout=60).raise_for_status()


# ------------------------------------------------------------
# ServeClass
# ------------------------------------------------------------


@app.cls(
    image=vllm_image,
    gpu=f"{config['service']['gpu']}:{N_GPU}",
    scaledown_window=config["service"]["scaledown_window"] * MINUTES,
    timeout=config["service"]["timeout"] * MINUTES,
    volumes=volumes,
    min_containers=config["service"]["min_containers"],
    max_containers=config["service"]["max_containers"],
    secrets=[
        modal.Secret.from_name(secret["name"]) for secret in config["service"]["secrets"]
    ],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=config["service"]["max_concurrent_requests"])
class VLLMServer:
    @modal.enter(snap=True)
    def start(self):
        # Runs BEFORE the snapshot is taken — only on a fresh cold start.
        # Steps: start vllm → wait ready → warmup (captures CUDA graphs) → sleep
        # After sleep() returns, Modal takes the GPU snapshot.
        # The snapshot contains: model weights (CPU), compiled kernels, CUDA graphs.
        import subprocess
        import os

        vllm_api_key = os.environ.get("VLLM_API_KEY")
        if not vllm_api_key:
            raise RuntimeError("Missing required VLLM_API_KEY environment variable.")

        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--uvicorn-log-level",
            "info",
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--trust-remote-code",
            "--api-key",
            vllm_api_key,
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--safetensors-load-strategy=prefetch",
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":2}',
            "--enable-sleep-mode",  # required for sleep/wake_up endpoints
            "--gpu-memory-utilization",
            str(GPU_MEMORY_UTILIZATION),
            "--max-cudagraph-capture-size",
            "32",  # limits CUDA graph capture batch size, reduces peak memory during capture
        ]

        if MAX_MODEL_LEN:
            cmd += ["--max-model-len", str(MAX_MODEL_LEN)]

        if MULTI_MODAL:
            cmd += ["--limit-mm-per-prompt", '{"image":1,"video":1}']
        else:
            cmd += ["--limit-mm-per-prompt", '{"image":0,"video":0}']

        # CUDA graphs are compiled once and captured in the snapshot — free on restore.
        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

        cmd += ["--tensor-parallel-size", str(N_GPU)]

        print(*cmd)
        self.proc = subprocess.Popen(cmd)

        _wait_ready(self.proc)
        print("vLLM ready — running warmup to capture CUDA graphs into snapshot")
        _warmup()
        print("Warmup done — putting vLLM to sleep before GPU snapshot")
        _sleep()
        print("vLLM sleeping — GPU snapshot will be taken now")

    @modal.enter(snap=False)
    def wake_up(self):
        # Runs on EVERY cold start including after snapshot restore.
        # Reloads model weights from CPU back to GPU.
        print("Restoring from snapshot — waking vLLM up")
        _wake_up()
        _wait_ready(self.proc)
        print("vLLM awake and ready")

    @modal.web_server(
        port=VLLM_PORT,
        startup_timeout=config["service"]["startup_timeout"] * MINUTES,
    )
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.proc.terminate()
