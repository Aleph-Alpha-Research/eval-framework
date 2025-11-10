import glob
import importlib
import importlib.metadata
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from eval_framework.run import run


@patch("argparse.ArgumentParser.parse_args")
@patch("eval_framework.response_generator.create_perturbation_class")
def test_run(mock_create_perturbation_class: Mock, mock_parse_args: Mock, tmp_path: Path) -> None:
    version_str = f"v{importlib.metadata.version('eval_framework')}"
    task_name = "ARC"
    llm_name = "SmolLM135M"
    mock_parse_args.return_value = Namespace(
        context="local",
        models=Path(__file__).parent / "conftest.py",
        llm_name=llm_name,
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
        llm_args=[],
        judge_models=Path(__file__).parent / "conftest.py",
        judge_model_name="Smollm135MInstruct",
        judge_model_args={},
        batch_size=2,
        task_subjects=None,
        description="",
        perturbation_type="editor",
        perturbation_probability=0.5,
        perturbation_seed=123,
        extra_task_modules=None,
        save_logs=True,
        resource_cleanup=True,
        delete_output_dir_after_upload=False,
    )

    mock_create_perturbation_class.side_effect = lambda x, _: x  # don't spin up docker here just for the test

    run()

    results_path = str(tmp_path / llm_name / f"{version_str}_{task_name}" / "*" / "results.jsonl")
    results_files = glob.glob(results_path)
    assert len(results_files) == 1


@patch("argparse.ArgumentParser.parse_args")
@patch("eval_framework.response_generator.create_perturbation_class")
def test_run_path(mock_create_perturbation_class: Mock, mock_parse_args: Mock, tmp_path: Path) -> None:
    version_str = f"v{importlib.metadata.version('eval_framework')}"
    task_name = "ARC"
    module = "tests.conftest"
    llm_name = "SmolLM135M"
    mock_parse_args.return_value = Namespace(
        context="local",
        models=None,
        llm_name=f"{module}.{llm_name}",
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
        llm_args=[],
        judge_models=Path(__file__).parent / "conftest.py",
        judge_model_name="tests.conftest.Smollm135MInstruct",
        judge_model_args={},
        batch_size=2,
        task_subjects=None,
        description="",
        perturbation_type="editor",
        perturbation_probability=0.5,
        perturbation_seed=123,
        extra_task_modules=None,
        save_logs=True,
        delete_output_dir_after_upload=False,
    )

    mock_create_perturbation_class.side_effect = lambda x, _: x  # don't spin up docker here just for the test

    run()

    results_path = str(tmp_path / llm_name / f"{version_str}_{task_name}" / "*" / "results.jsonl")
    results_files = glob.glob(results_path)
    assert len(results_files) == 1


@patch("argparse.ArgumentParser.parse_args")
def test_run_no_judge_model(mock_parse_args: Mock, tmp_path: Path) -> None:
    version_str = f"v{importlib.metadata.version('eval_framework')}"
    task_name = "ARC"
    llm_name = "SmolLM135M"
    mock_parse_args.return_value = Namespace(
        context="local",
        models=Path(__file__).parent / "conftest.py",
        llm_name=llm_name,
        num_samples=4,
        max_tokens=None,
        num_fewshot=0,
        task_name=task_name,
        hf_revision=None,
        output_dir=tmp_path,
        wandb_project="test-project",
        wandb_entity="test-entity",
        wandb_run_id="test-run",
        wandb_upload_results=True,
        hf_upload_dir="",
        hf_upload_repo="",
        llm_args=[],
        judge_models=None,
        judge_model_name=None,
        judge_model_args={},
        batch_size=2,
        task_subjects=None,
        description="",
        perturbation_type="",
        perturbation_probability=None,
        perturbation_seed=None,
        extra_task_modules=None,
        save_logs=True,
        delete_output_dir_after_upload=False,
    )

    run()

    results_path = str(tmp_path / llm_name / f"{version_str}_{task_name}" / "*" / "results.jsonl")
    results_files = glob.glob(results_path)
    assert len(results_files) == 1


@patch("argparse.ArgumentParser.parse_args")
def test_run_path_no_judge_model(mock_parse_args: Mock, tmp_path: Path) -> None:
    version_str = f"v{importlib.metadata.version('eval_framework')}"
    task_name = "ARC"
    module = "tests.conftest"
    llm_name = "SmolLM135M"
    mock_parse_args.return_value = Namespace(
        context="local",
        models=None,
        llm_name=f"{module}.{llm_name}",
        num_samples=4,
        max_tokens=None,
        num_fewshot=0,
        task_name=task_name,
        hf_revision=None,
        output_dir=tmp_path,
        wandb_project="test-project",
        wandb_entity="test-entity",
        wandb_run_id="test-run",
        wandb_upload_results=True,
        hf_upload_dir="",
        hf_upload_repo="",
        llm_args=[],
        judge_models=None,
        judge_model_name=None,
        judge_model_args={},
        batch_size=2,
        task_subjects=None,
        description="",
        perturbation_type="",
        perturbation_probability=None,
        perturbation_seed=None,
        extra_task_modules=None,
        save_logs=True,
        delete_output_dir_after_upload=False,
    )

    run()

    results_path = str(tmp_path / llm_name / f"{version_str}_{task_name}" / "*" / "results.jsonl")
    results_files = glob.glob(results_path)
    assert len(results_files) == 1
