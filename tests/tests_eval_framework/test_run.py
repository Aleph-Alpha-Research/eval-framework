import glob
import importlib.metadata
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from eval_framework.llm.openai import OpenAIModel
from eval_framework.run import _build_config, run, run_with_kwargs
from eval_framework.utils.llama_server import LLAMA_SERVER_BASE_URL
from tests.tests_eval_framework.conftest import MockLLM


@pytest.fixture(autouse=True)
def llama_server() -> Iterator[Mock]:
    """No test may launch a real llama-server."""
    with patch("eval_framework.run.llama_server") as mock:
        yield mock


class NamedMockLLM(MockLLM):
    """Stands in for OpenAIModel, which always receives a model_name."""

    def __init__(self, model_name: str) -> None:
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


def test_run_with_kwargs_stays_remote_without_self_host_flag(llama_server: Mock) -> None:
    with patch("eval_framework.run._run_single_task") as run_single_task:
        run_with_kwargs({"llm_args": {"model_name": "m"}})

    assert run_single_task.call_args.args[0] == {"llm_args": {"model_name": "m"}}
    llama_server.assert_not_called()


def test_run_with_kwargs_self_host_flag_starts_and_stops_the_server(llama_server: Mock) -> None:
    with patch("eval_framework.run._run_single_task") as run_single_task:
        run_with_kwargs({"llm_args": {"model_name": "org/model-GGUF"}, "self_host": True, "output_dir": "out"})

    assert run_single_task.call_args.args[0]["llm_args"] == {
        "model_name": "org/model-GGUF",
        "base_url": LLAMA_SERVER_BASE_URL,
        "api_key": "no-key",
    }
    assert "self_host" not in run_single_task.call_args.args[0]
    llama_server.assert_called_once_with("org/model-GGUF", Path("out"))
    llama_server.return_value.__enter__.assert_called_once()
    llama_server.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"llm_args": {}},
        {"llm_args": {"model_name": "m", "base_url": "http://other/v1"}},
        {"llm_args": {"model_name": "m"}, "task_suite": "suite.yaml"},
    ],
)
def test_run_with_kwargs_self_host_flag_rejects_conflicting_arguments(
    llama_server: Mock, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="--self-host"):
        run_with_kwargs({**kwargs, "self_host": True})

    llama_server.assert_not_called()
