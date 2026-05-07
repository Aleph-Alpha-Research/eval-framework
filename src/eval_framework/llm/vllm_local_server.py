from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

from eval_framework.llm.base import BaseLLM
from eval_framework.llm.openai import OpenAIModel
from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import BaseFormatter, Message

logger = logging.getLogger(__name__)


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_http_ready(url: str, *, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for server readiness at {url}. Last error: {last_err}")


def _wait_for_http_ready_or_proc_exit(url: str, *, timeout_s: float, proc: subprocess.Popen[str]) -> None:
    """
    Like `_wait_for_http_ready`, but fail fast if the server process exits.

    This avoids long timeouts that hide the real root cause (e.g. invalid CLI flags,
    missing dependencies, CUDA issues).
    """
    deadline = time.time() + timeout_s
    last_err: Exception | None = None

    while time.time() < deadline:
        if proc.poll() is not None:
            out = ""
            try:
                if proc.stdout is not None:
                    out = proc.stdout.read() or ""
            except Exception:
                out = ""
            tail = out.strip()
            if len(tail) > 8000:
                tail = tail[-8000:]
            raise RuntimeError(
                f"vLLM server process exited before becoming ready. exit_code={proc.returncode}. Output (tail):\n{tail}"
            )

        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
            time.sleep(0.25)

    raise RuntimeError(f"Timed out waiting for server readiness at {url}. Last error: {last_err}")


class VLLMLocalServerOpenAIModel(BaseLLM):
    """
    Provider-style model: start a local vLLM OpenAI-compatible server, then talk to it via `OpenAIModel(base_url=...)`.

    This gives you a stable HTTP boundary (good for VCR cassettes) while keeping "local vLLM" as a selectable backend.

    Notes:
    - The server is started in a subprocess using `vllm serve`.
    - Cleanup is best-effort (SIGTERM then SIGKILL).
    - Not all OpenAI API features are guaranteed to be supported by the local server (e.g. logprobs).
    """

    def __init__(
        self,
        *,
        model_name: str,
        host: str = "127.0.0.1",
        port: int | None = None,
        startup_timeout_s: float = 120.0,
        # `OpenAIModel` parameters:
        formatter: BaseFormatter | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        api_key: str | None = None,
        bytes_per_token: float | None = None,
        # vLLM "serve" parameters (subset, passed through):
        tensor_parallel_size: int | None = None,
        dtype: str | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        enforce_eager: bool | None = None,
        # Escape hatch:
        vllm_command: str | None = None,
        vllm_extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._host = host
        self._port = port if port is not None else _pick_free_port(host)
        self._startup_timeout_s = float(startup_timeout_s)

        self._proc: subprocess.Popen[str] | None = None

        self._server_url = f"http://{self._host}:{self._port}/v1"

        cmd = [vllm_command or "vllm", "serve", self._model_name, "--host", self._host, "--port", str(self._port)]

        # A small, intentionally conservative subset of flags.
        if tensor_parallel_size is not None:
            cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]
        if dtype is not None:
            cmd += ["--dtype", str(dtype)]
        if max_model_len is not None:
            cmd += ["--max-model-len", str(max_model_len)]
        if gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
        if enforce_eager is not None:
            # vLLM exposes this as a boolean flag; passing a value breaks CLI parsing.
            if enforce_eager:
                cmd += ["--enforce-eager"]

        if vllm_extra_args:
            cmd += list(vllm_extra_args)

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        logger.info("Starting local vLLM server: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Ensure we don't leave it around if the process exits abruptly.
        atexit.register(self._cleanup)

        # Wait until the OpenAI-compatible endpoints respond.
        if self._proc is None:
            raise RuntimeError("Failed to start vLLM server process.")
        _wait_for_http_ready_or_proc_exit(
            f"{self._server_url}/models",
            timeout_s=self._startup_timeout_s,
            proc=self._proc,
        )

        # Configure client to talk to the local server.
        # For local servers, any non-empty API key typically works; allow explicit override.
        effective_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY") or "local-vllm"

        self._client = OpenAIModel(
            model_name=self._model_name,
            formatter=formatter,
            temperature=temperature,
            top_p=top_p,
            api_key=effective_api_key,
            base_url=self._server_url,
            bytes_per_token=bytes_per_token,
        )

    @property
    def name(self) -> str:
        return f"vllm_local::{self._model_name}"

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[RawCompletion]:
        return self._client.generate_from_messages(messages, stop_sequences, max_tokens, temperature, top_p)

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        return self._client.logprobs(samples)

    def _cleanup(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is not None:
            return

        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.send_signal(signal.SIGKILL)
            except Exception:
                pass

    def __del__(self) -> None:
        self._cleanup()
