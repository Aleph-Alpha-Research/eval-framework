import glob
import importlib.metadata
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from eval_framework.llm.openai import OpenAIModel
from eval_framework.run import _build_config, run
from tests.tests_eval_framework.conftest import MockLLM


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
