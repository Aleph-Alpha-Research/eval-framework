import glob
import importlib
import importlib.metadata
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from eval_framework.run import run
from tests.tests_eval_framework.conftest import MockLLM


class NamedMockLLM(MockLLM):
    """Stands in for OpenAIModel, which always receives a model_name."""

    def __init__(self, model_name: str) -> None:
        super().__init__()


def _namespace(tmp_path: Path, task_name: str, judge_model_args: list[str]) -> Namespace:
    return Namespace(
        context="local",
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
        resource_cleanup=True,
        delete_output_dir_after_upload=False,
        randomize_judge_order=False,
        repeats=1,
        task_suite=None,
    )


@patch("eval_framework.context.local.OpenAIModel", NamedMockLLM)
@patch("argparse.ArgumentParser.parse_args")
def test_run(mock_parse_args: Mock, tmp_path: Path) -> None:
    version_str = f"v{importlib.metadata.version('eval_framework')}"
    task_name = "ARC"
    mock_parse_args.return_value = _namespace(tmp_path, task_name, judge_model_args=["model_name=judge"])

    run()

    results_path = str(tmp_path / "NamedMockLLM" / f"{version_str}_{task_name}" / "*" / "results.jsonl")
    results_files = glob.glob(results_path)
    assert len(results_files) == 1


@patch("eval_framework.context.local.OpenAIModel", NamedMockLLM)
@patch("argparse.ArgumentParser.parse_args")
def test_run_no_judge_model(mock_parse_args: Mock, tmp_path: Path) -> None:
    version_str = f"v{importlib.metadata.version('eval_framework')}"
    task_name = "ARC"
    mock_parse_args.return_value = _namespace(tmp_path, task_name, judge_model_args=[])

    run()

    results_path = str(tmp_path / "NamedMockLLM" / f"{version_str}_{task_name}" / "*" / "results.jsonl")
    results_files = glob.glob(results_path)
    assert len(results_files) == 1
