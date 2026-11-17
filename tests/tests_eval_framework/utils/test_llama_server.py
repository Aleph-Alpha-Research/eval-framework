import socket
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from eval_framework.utils.llama_server import _port_in_use, llama_server

MODULE = "eval_framework.utils.llama_server"


def test_port_in_use_tells_a_listening_port_from_a_free_one() -> None:
    with socket.create_server(("localhost", 0)) as listener:
        port = listener.getsockname()[1]
        assert _port_in_use(port)

    assert not _port_in_use(port)


def test_llama_server_refuses_a_busy_port(tmp_path: Path) -> None:
    with (
        patch(f"{MODULE}._port_in_use", return_value=True),
        patch(f"{MODULE}.subprocess.Popen") as popen,
        pytest.raises(RuntimeError, match="already in use"),
        llama_server("m", tmp_path),
    ):
        pass

    popen.assert_not_called()


def test_llama_server_explains_a_missing_llama_server(tmp_path: Path) -> None:
    with (
        patch(f"{MODULE}._port_in_use", return_value=False),
        patch(f"{MODULE}.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="brew install llama.cpp"),
        llama_server("m", tmp_path),
    ):
        pass


def test_llama_server_launches_llama_server_and_stops_it_after_the_block(tmp_path: Path) -> None:
    with (
        patch(f"{MODULE}._port_in_use", return_value=False),
        patch(f"{MODULE}.shutil.which", return_value="/bin/llama-server"),
        patch(f"{MODULE}.subprocess.Popen") as popen,
        patch(f"{MODULE}.urllib.request.urlopen", side_effect=[OSError, Mock()]),
        patch(f"{MODULE}.time.sleep"),
    ):
        popen.return_value.poll.return_value = None

        with llama_server("org/model-GGUF", tmp_path):
            popen.return_value.kill.assert_not_called()

    assert popen.call_args.args[0] == ["llama-server", "-hf", "org/model-GGUF", "--port", "8080"]
    popen.return_value.kill.assert_called_once()
    popen.return_value.wait.assert_called_once()


def test_llama_server_stops_the_server_when_interrupted_during_startup(tmp_path: Path) -> None:
    with (
        patch(f"{MODULE}._port_in_use", return_value=False),
        patch(f"{MODULE}.shutil.which", return_value="/bin/llama-server"),
        patch(f"{MODULE}.subprocess.Popen") as popen,
        patch(f"{MODULE}.urllib.request.urlopen", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        popen.return_value.poll.return_value = None
        with llama_server("org/model-GGUF", tmp_path):
            pass

    popen.return_value.kill.assert_called_once()
    popen.return_value.wait.assert_called_once()
