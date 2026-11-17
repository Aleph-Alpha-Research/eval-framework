"""Serve a model with llama.cpp's llama-server for the duration of a run."""

import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# llama-server listens here by default.
LLAMA_SERVER_PORT = 8080
LLAMA_SERVER_BASE_URL = f"http://localhost:{LLAMA_SERVER_PORT}/v1"


def _port_in_use(port: int) -> bool:
    try:
        socket.create_connection(("localhost", port), timeout=1).close()
    except OSError:  # nothing accepted the connection
        return False
    return True


def _serves_alias(port: int, alias: str) -> bool:
    """Whether the server on `port` lists a model named `alias`."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2) as response:
            return alias in response.read().decode(errors="replace")
    except OSError:
        return False


def _wait_until_healthy(server: subprocess.Popen, alias: str, model: str, log_path: Path) -> None:
    # No timeout: the download can take long. A failed start ends the process, which ends the wait.
    while server.poll() is None:
        try:
            urllib.request.urlopen(f"http://localhost:{LLAMA_SERVER_PORT}/health", timeout=2).close()
        except (urllib.error.URLError, OSError):  # not listening yet, or 503 while the model loads
            time.sleep(1)
            continue
        # llama-server only listens once the model is loaded, so the port stays open to others
        # for the whole startup, long after it was found free.
        if not _serves_alias(LLAMA_SERVER_PORT, alias):
            raise RuntimeError(
                f"Another server took port {LLAMA_SERVER_PORT} while llama-server was starting {model}. "
                "Its answers would be recorded under the wrong model, so --self-host stops here."
            )
        return
    raise RuntimeError(f"llama-server exited while starting {model}; see {log_path}")


@contextmanager
def llama_server(model: str, output_dir: Path) -> Iterator[None]:
    """Serve `model` (a Hugging Face GGUF repo) at `LLAMA_SERVER_BASE_URL` until the block exits."""
    if _port_in_use(LLAMA_SERVER_PORT):
        # Never reuse it: another model could answer and be recorded under `model`.
        raise RuntimeError(
            f"Port {LLAMA_SERVER_PORT} is already in use, so --self-host cannot start llama-server. Stop that "
            "process, or drop --self-host and pass `base_url=...` in --llm-args to use your own server."
        )
    if shutil.which("llama-server") is None:
        raise RuntimeError(
            "--self-host needs `llama-server` on the PATH. Install llama.cpp (`brew install llama.cpp`)."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "llama-server.log"
    print(f"Starting llama-server for {model} (first run downloads the model; log: {log_path})")
    # Unique per run, so the server that answers can be told apart from anyone else's.
    alias = f"eval-framework-{uuid.uuid4().hex}"
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            ["llama-server", "-hf", model, "--port", str(LLAMA_SERVER_PORT), "--alias", alias],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        _wait_until_healthy(server, alias, model, log_path)
        yield
    finally:  # also reached on Ctrl-C during the download
        server.kill()  # terminate() is ignored while the server is generating
        server.wait()
