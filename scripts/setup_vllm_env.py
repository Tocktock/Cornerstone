#!/usr/bin/env python3
"""
Set up a local vLLM OpenAI-compatible endpoint for Cornerstone.

This script works across macOS, Windows, and Linux. It prefers a locally
available `vllm` CLI (installed via pip) and falls back to Docker
(`vllm/vllm-openai`) when the CLI is unavailable. It also writes the
environment variables Cornerstone needs to target vLLM.

Defaults (4-bit quantised models):
  - Chat:      OpenAccess-AI-Collective/gpt-oss-20b-gptq-4bit -> served as `gpt-oss-20b`
  - Embedding: Qwen/Qwen2-7B-Embedding-GPTQ-Int4             -> served as `qwen3-embedding-4b`

Usage
-----
  python scripts/setup_vllm_env.py              # pull models, start server (if needed)
  python scripts/setup_vllm_env.py --env-file .env.vllm
  python scripts/setup_vllm_env.py status       # inspect current state
  python scripts/setup_vllm_env.py stop         # stop docker container (if docker fallback)
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence
from urllib.error import URLError
from urllib.request import urlopen

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

DEFAULT_CHAT_MODEL = "OpenAccess-AI-Collective/gpt-oss-20b-gptq-4bit"
DEFAULT_CHAT_ALIAS = "gpt-oss-20b"
DEFAULT_CHAT_QUANT = "gptq"

DEFAULT_EMBED_MODEL = "Qwen/Qwen2-7B-Embedding-GPTQ-Int4"
DEFAULT_EMBED_ALIAS = "qwen3-embedding-4b"
DEFAULT_EMBED_QUANT = "gptq"

DEFAULT_PORT = 8000
CONTAINER_NAME = "cornerstone-vllm"
VOLUME_NAME = "cornerstone-vllm-hf-cache"
VLLM_IMAGE = "vllm/vllm-openai:latest"

ENV_TEMPLATE = {
    "CHAT_BACKEND": "vllm",
    "VLLM_BASE_URL": "http://localhost:{port}",
    "VLLM_MODEL": DEFAULT_CHAT_ALIAS,
    "EMBEDDING_MODEL": f"vllm:{DEFAULT_EMBED_ALIAS}",
    "VLLM_EMBEDDING_BASE_URL": "http://localhost:{port}",
}

# --------------------------------------------------------------------------- #
# Helper utilities                                                             #
# --------------------------------------------------------------------------- #


def run_command(
    command: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """Execute a command with uniform error handling."""
    try:
        return subprocess.run(command, check=check, capture_output=capture_output, text=text)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout.strip() if exc.stdout else ""
        stderr = exc.stderr.strip() if exc.stderr else ""
        details = "\n".join(filter(None, [stdout, stderr]))
        raise RuntimeError(f"Command failed ({' '.join(command)}):\n{details}") from exc


def have_executable(name: str) -> bool:
    return shutil.which(name) is not None


def gpu_available() -> bool:
    """Return True if an NVIDIA GPU is accessible on the host."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------- #
# vLLM via local CLI                                                           #
# --------------------------------------------------------------------------- #


def have_local_vllm() -> bool:
    return have_executable("vllm")


def ensure_local_vllm_server(
    chat_model: str,
    chat_alias: str,
    chat_quant: str,
    embed_model: str,
    embed_alias: str,
    embed_quant: str,
    port: int,
    env: dict[str, str],
) -> None:
    """
    Launch vLLM via the local CLI if a server is not already listening.

    We attempt to detect an existing server by probing the OpenAI `/models`
    endpoint. If none is running, we spawn a detached `vllm serve` process.
    """
    if server_is_ready(env["VLLM_BASE_URL"]):
        print("[vllm-cli] Existing server detected; skipping launch.")
        return

    if not gpu_available():
        raise RuntimeError(
            "No NVIDIA GPU detected. vLLM requires CUDA-capable GPUs; "
            "install GPU drivers or use a GPU-enabled machine."
        )

    command = [
        "vllm",
        "serve",
        chat_model,
        "--served-model-name",
        chat_alias,
        "--quantization",
        chat_quant,
        "--model",
        embed_model,
        "--served-model-name",
        embed_alias,
        "--quantization",
        embed_quant,
        "--port",
        str(port),
    ]

    # vLLM currently requires CUDA; warn if env lacks GPUs.
    print(f"[vllm-cli] Launching: {' '.join(shlex.quote(part) for part in command)}")
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    wait_for_server(env["VLLM_BASE_URL"])


# --------------------------------------------------------------------------- #
# vLLM via Docker                                                              #
# --------------------------------------------------------------------------- #


def have_docker() -> bool:
    return have_executable("docker")


def docker_container_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def docker_container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    return result.stdout.strip().lower() == "true"


def ensure_docker_volume(name: str) -> None:
    result = subprocess.run(
        ["docker", "volume", "inspect", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        print(f"[docker] Creating volume {name} ...")
        run_command(["docker", "volume", "create", name])


def ensure_docker_runtime(
    chat_model: str,
    chat_alias: str,
    chat_quant: str,
    embed_model: str,
    embed_alias: str,
    embed_quant: str,
    port: int,
) -> None:
    if not have_docker():
        raise RuntimeError(
            "Docker CLI not found. Install Docker Desktop (Windows/macOS) or Docker Engine (Linux)."
        )

    if not gpu_available():
        raise RuntimeError(
            "No NVIDIA GPU detected on the host. vLLM requires CUDA-capable GPUs. "
            "If you intended to use CPU-only inference, consider an alternative runtime."
        )

    ensure_docker_volume(VOLUME_NAME)

    hf_cache = resolve_hf_cache()
    os.makedirs(hf_cache, exist_ok=True)

    base_command = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "--restart",
        "unless-stopped",
        "-p",
        f"{port}:8000",
        "-v",
        f"{hf_cache}:{'/root/.cache/huggingface'}",
        "-e",
        f"HF_HOME={Path('/root/.cache/huggingface')}",
    ]

    if gpu_available():
        base_command.extend(["--gpus", "all"])

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        base_command.extend(["-e", f"HUGGINGFACE_HUB_TOKEN={hf_token}"])

    launch_command = [
        VLLM_IMAGE,
        "--model",
        chat_model,
        "--served-model-name",
        chat_alias,
        "--quantization",
        chat_quant,
        "--model",
        embed_model,
        "--served-model-name",
        embed_alias,
        "--quantization",
        embed_quant,
    ]

    if docker_container_exists(CONTAINER_NAME):
        if docker_container_running(CONTAINER_NAME):
            print(f"[docker] Container {CONTAINER_NAME} already running.")
        else:
            print(f"[docker] Starting existing container {CONTAINER_NAME} ...")
            run_command(["docker", "start", CONTAINER_NAME])
            wait_for_server(f"http://localhost:{port}")
        return

    print(f"[docker] Launching container {CONTAINER_NAME} ...")
    cmd = base_command + launch_command
    print(f"[docker] {' '.join(shlex.quote(part) for part in cmd)}")
    run_command(cmd)
    wait_for_server(f"http://localhost:{port}")


def stop_docker_container() -> None:
    if docker_container_exists(CONTAINER_NAME):
        print(f"[docker] Stopping container {CONTAINER_NAME} ...")
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER_NAME],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print(f"[docker] No container named {CONTAINER_NAME}.")


# --------------------------------------------------------------------------- #
# Server health checks                                                         #
# --------------------------------------------------------------------------- #


def server_is_ready(base_url: str) -> bool:
    try:
        with urlopen(f"{base_url}/v1/models", timeout=3) as resp:  # type: ignore[call-arg]
            return 200 <= resp.status < 300
    except URLError:
        return False


def wait_for_server(base_url: str, timeout: int = 180) -> None:
    print("[setup] Waiting for vLLM server to become ready ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_is_ready(base_url):
            print("[setup] vLLM server is ready.")
            return
        time.sleep(4)
    raise RuntimeError("Timed out waiting for vLLM server. Check logs for details.")


# --------------------------------------------------------------------------- #
# Environment helpers                                                          #
# --------------------------------------------------------------------------- #


def resolve_hf_cache() -> str:
    env_home = os.environ.get("HF_HOME")
    if env_home:
        return str(Path(env_home).expanduser().resolve())
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return str(Path(xdg_cache).expanduser().resolve() / "huggingface")
    return str(Path.home() / ".cache" / "huggingface")


def write_env_file(path: Path, env: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for key, value in env.items():
            fh.write(f"{key}={value}\n")
    print(f"[setup] Wrote environment variables to {path}")


def print_env_exports(env: dict[str, str]) -> None:
    print("\nExport these variables before launching Cornerstone:\n")
    for key, value in env.items():
        print(f"  {key}={value}")
    print()


# --------------------------------------------------------------------------- #
# Status reporting                                                             #
# --------------------------------------------------------------------------- #


def show_status(env: dict[str, str]) -> None:
    print("vLLM CLI available :", "yes" if have_local_vllm() else "no")
    print("Docker available   :", "yes" if have_docker() else "no")
    if docker_container_exists(CONTAINER_NAME):
        state = "running" if docker_container_running(CONTAINER_NAME) else "stopped"
        print(f"Docker container   : {CONTAINER_NAME} ({state})")
    else:
        print("Docker container   : not created")
    print("Suggested env vars :")
    for key, value in env.items():
        print(f"  {key}={value}")


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #


def start(args: argparse.Namespace) -> None:
    chat_model = args.chat_model or DEFAULT_CHAT_MODEL
    chat_alias = args.chat_alias or DEFAULT_CHAT_ALIAS
    chat_quant = args.chat_quant or DEFAULT_CHAT_QUANT
    embed_model = args.embedding_model or DEFAULT_EMBED_MODEL
    embed_alias = args.embed_alias or DEFAULT_EMBED_ALIAS
    embed_quant = args.embed_quant or DEFAULT_EMBED_QUANT
    port = args.port

    env = {
        "CHAT_BACKEND": "vllm",
        "VLLM_BASE_URL": f"http://localhost:{port}",
        "VLLM_MODEL": chat_alias,
        "EMBEDDING_MODEL": f"vllm:{embed_alias}",
        "VLLM_EMBEDDING_BASE_URL": f"http://localhost:{port}",
    }

    if have_local_vllm():
        ensure_local_vllm_server(
            chat_model,
            chat_alias,
            chat_quant,
            embed_model,
            embed_alias,
            embed_quant,
            port,
            env,
        )
    else:
        ensure_docker_runtime(
            chat_model,
            chat_alias,
            chat_quant,
            embed_model,
            embed_alias,
            embed_quant,
            port,
        )

    print_env_exports(env)
    if args.env_file:
        write_env_file(Path(args.env_file).expanduser(), env)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a vLLM server for Cornerstone.")
    parser.add_argument(
        "action",
        choices=("start", "stop", "status"),
        nargs="?",
        default="start",
        help="start (default) to launch/ensure server, stop to remove docker container, status to inspect",
    )
    parser.add_argument("--env-file", help="Optional .env output path.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to expose (default 8000).")
    parser.add_argument("--chat-model", help="HuggingFace ID or local path for the chat model.")
    parser.add_argument("--chat-alias", help="Served name for the chat model (default gpt-oss-20b).")
    parser.add_argument("--chat-quant", help="Quantization type for the chat model (default gptq).")
    parser.add_argument("--embedding-model", help="HuggingFace ID or local path for the embedding model.")
    parser.add_argument("--embed-alias", help="Served name for the embedding model (default qwen3-embedding-4b).")
    parser.add_argument("--embed-quant", help="Quantization type for the embedding model (default gptq).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    env = {
        "CHAT_BACKEND": "vllm",
        "VLLM_BASE_URL": f"http://localhost:{args.port}",
        "VLLM_MODEL": args.chat_alias or DEFAULT_CHAT_ALIAS,
        "EMBEDDING_MODEL": f"vllm:{args.embed_alias or DEFAULT_EMBED_ALIAS}",
        "VLLM_EMBEDDING_BASE_URL": f"http://localhost:{args.port}",
    }

    try:
        if args.action == "start":
            start(args)
        elif args.action == "stop":
            stop_docker_container()
        elif args.action == "status":
            show_status(env)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    sys.exit(main())
