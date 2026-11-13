import glob
import importlib.metadata
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from eval_framework.llm.openai import OpenAIModel
from eval_framework.run import LOCAL_BASE_URL, _build_config, _start_local_server, run, run_with_kwargs
from tests.tests_eval_framework.conftest import MockLLM


@pytest.fixture(autouse=True)
def start_local_server() -> Iterator[Mock]:
    """No test may launch a real llama-server."""
    with patch("eval_framework.run._start_local_server", return_value=None) as mock:
        yield mock


class NamedMockLLM(MockLLM):
    """Stands in for OpenAIModel, which always receives a model_name."""

    def __init__(self, model_name: str, base_url: str | None = None) -> None:
        super().__init__()


def _namespace(tmp_path: Path, task_name: str, judge_model_args: list[str]) -> Namespace:
    return Namespace(
        num_samples=4,
        max_tokens=None,
        num_fewshot=0,
        task_name=task_name,
        hf_revision=None,
        wandb_project="test-project",
        wandb_entity="test-entity",
        wandb_run_id="test-run",
        wandb_upload_results=True,
        output_dir=tmp_path,
        hf_upload_dir="",
        hf_upload_repo="",
        llm_args=["model_name=mock"],
        judge_model_args=judge_model_args,
        batch_size=2,
        task_subjects=None,
        description="",
        extra_tasks_dir=None,
        save_logs=True,
        delete_output_dir_after_upload=False,
        randomize_judge_order=False,
        repeats=1,
        task_suite=None,
    )


@patch("eval_framework.run.OpenAIModel", NamedMockLLM)
@patch("argparse.ArgumentParser.parse_args")
def test_run(mock_parse_args: Mock, tmp_path: Path) -> None:
    version_str = f"v{importlib.metadata.version('eval_framework')}"
    task_name = "ARC"
    mock_parse_args.return_value = _namespace(tmp_path, task_name, judge_model_args=["model_name=judge"])

    run()

    results_path = str(tmp_path / "NamedMockLLM" / f"{version_str}_{task_name}" / "*" / "results.jsonl")
    results_files = glob.glob(results_path)
    assert len(results_files) == 1


@patch("eval_framework.run.OpenAIModel", NamedMockLLM)
@patch("argparse.ArgumentParser.parse_args")
def test_run_no_judge_model(mock_parse_args: Mock, tmp_path: Path) -> None:
    version_str = f"v{importlib.metadata.version('eval_framework')}"
    task_name = "ARC"
    mock_parse_args.return_value = _namespace(tmp_path, task_name, judge_model_args=[])

    run()

    results_path = str(tmp_path / "NamedMockLLM" / f"{version_str}_{task_name}" / "*" / "results.jsonl")
    results_files = glob.glob(results_path)
    assert len(results_files) == 1


def _kwargs(llm_args: dict[str, str], judge_model_args: dict[str, str]) -> dict:
    return {
        **vars(_namespace(Path("dummy"), "ARC", judge_model_args=[])),
        "llm_args": llm_args,
        "judge_model_args": judge_model_args,
        "repeats": 1,
    }


def test_build_config_with_judge() -> None:
    config = _build_config(_kwargs({"model_name": "m"}, {"model_name": "judge"}))

    assert config.llm_class is OpenAIModel
    assert config.llm_args == {"model_name": "m"}
    assert config.llm_judge_class is OpenAIModel
    assert config.judge_model_args == {"model_name": "judge"}
    assert config.task_name == "ARC"
    assert config.repeats == 1


def test_build_config_without_judge() -> None:
    config = _build_config(_kwargs({"model_name": "m"}, {}))

    assert config.llm_judge_class is None


@pytest.mark.parametrize(
    ("llm_args", "judge_model_args", "match"),
    [
        ({}, {}, "llm_args must set model_name"),
        ({"model_name": "m"}, {"temperature": "0"}, "judge_model_args must set model_name"),
    ],
)
def test_build_config_requires_model_names(
    llm_args: dict[str, str], judge_model_args: dict[str, str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _build_config(_kwargs(llm_args, judge_model_args))


@pytest.mark.parametrize(
    ("env", "llm_args", "expected"),
    [
        ({}, {"model_name": "m"}, LOCAL_BASE_URL),
        ({"OPENAI_API_KEY": "k"}, {"model_name": "m"}, None),
        ({"OPENAI_BASE_URL": "http://env/v1"}, {"model_name": "m"}, None),
        ({}, {"model_name": "m", "base_url": "http://other/v1"}, "http://other/v1"),
    ],
)
def test_run_with_kwargs_defaults_to_local_server(
    monkeypatch: pytest.MonkeyPatch,
    start_local_server: Mock,
    env: dict[str, str],
    llm_args: dict[str, str],
    expected: str | None,
) -> None:
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)

    with patch("eval_framework.run._run_single_task") as run_single_task:
        run_with_kwargs({"llm_args": llm_args})

    assert run_single_task.call_args.args[0]["llm_args"].get("base_url") == expected
    assert start_local_server.called == (expected == LOCAL_BASE_URL)


def test_start_local_server_reuses_a_running_server(tmp_path: Path) -> None:
    with patch("eval_framework.run.socket.create_connection"), patch("eval_framework.run.subprocess.Popen") as popen:
        assert _start_local_server("m", tmp_path) is None

    popen.assert_not_called()


def test_start_local_server_explains_a_missing_llama_server(tmp_path: Path) -> None:
    with (
        patch("eval_framework.run.socket.create_connection", side_effect=OSError),
        patch("eval_framework.run.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="brew install llama.cpp"),
    ):
        _start_local_server("m", tmp_path)


def test_start_local_server_launches_llama_server_and_waits_for_health(tmp_path: Path) -> None:
    with (
        patch("eval_framework.run.socket.create_connection", side_effect=OSError),
        patch("eval_framework.run.shutil.which", return_value="/bin/llama-server"),
        patch("eval_framework.run.subprocess.Popen") as popen,
        patch("eval_framework.run.urllib.request.urlopen", side_effect=[OSError, Mock()]),
        patch("eval_framework.run.time.sleep"),
    ):
        popen.return_value.poll.return_value = None

        assert _start_local_server("org/model-GGUF", tmp_path) is popen.return_value

    assert popen.call_args.args[0] == ["llama-server", "-hf", "org/model-GGUF", "--port", "8080"]
