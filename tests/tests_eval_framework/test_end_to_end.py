import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from datasets import disable_caching

from eval_framework.llm.base import BaseLLM
from eval_framework.main import main
from eval_framework.result_processors.base import Result
from eval_framework.tasks.benchmarks.hellaswag import HELLASWAG
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.task_names import TaskNameEnum
from eval_framework.utils.constants import GREEN, RED, RESET
from tests.tests_eval_framework.conftest import MockLLM


def pretty_print_results(results: list[Result]) -> None:
    for result in results:
        print(f"{RED}{result.llm_name} | {result.metric_name}: {result.value} | {result.subject}{RESET}")


def pretty_print_means(metric_means: dict[str, float], expected_results: dict[str, float]) -> None:
    for metric_name, metric_mean in metric_means.items():
        status = GREEN if metric_mean == expected_results[metric_name] else RED
        print(f"{metric_name}: {status}{metric_mean}{RESET} - Expected: {expected_results[metric_name]}")


def _almost_equal(x: float, y: float) -> bool:
    return 2 * abs(x - y) / abs(x + y + 1e-5) < 1e-4


@pytest.mark.gpu
@pytest.mark.parametrize("test_llms", ["SmolLM135M"], indirect=["test_llms"])
def test_automatic_tasks(tmp_path: Path, test_llms: BaseLLM) -> None:
    output_dir = tmp_path / "eval"

    task_name = HELLASWAG.NAME
    expected_results = {
        "Accuracy Loglikelihood": 0.4,
        "Accuracy Normalized Loglikelihood": 0.6,
        "Accuracy Bayesian Loglikelihood": 0.5,
    }
    num_fewshot = 2
    num_samples = 10

    eval_config = EvalConfig(
        task_name=task_name,
        num_fewshot=num_fewshot,
        num_samples=num_samples,
        output_dir=output_dir,
        llm_class=test_llms.__class__,
        save_intermediate_results=False,
        task_subjects=None,
    )

    results = main(test_llms, eval_config)

    full_metric_names = [
        (result.metric_name, result.key) for result in results if result.metric_name in list(expected_results.keys())
    ]
    metric_means = {}
    for metric_name, key in sorted(set(full_metric_names)):
        values = [
            result.value
            for result in results
            if result.metric_name == metric_name and result.key == key and result.value is not None
        ]
        metric_means[metric_name] = 0 if len(values) == 0 else sum(values) / len(values)

    pretty_print_means(metric_means, expected_results)

    for metric_name, metric_mean in metric_means.items():
        if key is not None:
            metric_name += f"/{key}"
        print(f"Expect {metric_name} to be {expected_results[metric_name]} and got {metric_mean}")
        assert _almost_equal(expected_results[metric_name], metric_mean), (
            f"Expected {metric_name} to be {expected_results[metric_name]} but got {metric_mean}"
        )


def pytest_generate_tests(metafunc: Any) -> None:
    # dynamic parameterization for test_all_tasks_with_all_samples: go over all tasks
    if "full_task_name" in metafunc.fixturenames:
        task_names = [task.value for task in TaskNameEnum]
        ids = [task.name for task in TaskNameEnum]

        metafunc.parametrize("full_task_name", task_names, ids=ids)


logger = logging.getLogger(__name__)


def get_disk_usage_info() -> str:
    """Get disk usage information for the root directory."""
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)

    # Get top directories by size
    try:
        # Get the top 5 largest directories
        top_dirs_result = subprocess.run(
            "du -h --max-depth=3 / 2>/dev/null | sort -hr | head -5", shell=True, capture_output=True, text=True
        )

        # Get the specific directories you're interested in
        specific_dirs_result = subprocess.run(
            "du -h /tmp/pytest-of-runner/pytest-0 2>/dev/null", shell=True, capture_output=True, text=True
        )

        top_dirs = top_dirs_result.stdout
        specific_dirs = specific_dirs_result.stdout
    except Exception:
        top_dirs = "Could not get directory sizes"
        specific_dirs = "Could not get directory sizes"

    return f"Free space: {free_gb:.2f}GB\nTop directories:\n{top_dirs}\nSpecific directories:\n{specific_dirs}\n"


@pytest.mark.cpu_slow
@patch("eval_framework.tasks.utils.run_python_code")
def test_with_all_samples(mock_run_python_code: Mock, tmp_path: Path, full_task_name: str) -> None:
    """Smoke test with disk space cleanup."""
    # Disable HF dataset caching before the test
    disable_caching()

    # Your existing test code
    output_dir = tmp_path / "eval"
    num_samples = None

    eval_config = EvalConfig(
        task_name=full_task_name,
        num_fewshot=0,
        num_samples=num_samples,
        output_dir=output_dir,
        llm_class=MockLLM,
        llm_judge_class=MockLLM,
        save_intermediate_results=False,
    )

    mock_run_python_code.return_value = ""

    try:
        main(MockLLM(), eval_config)
        print(f"DISK SPACE AFTER TEST: {get_disk_usage_info()}")
        logger.info(f"DISK SPACE AFTER TEST: {get_disk_usage_info()}")
    finally:
        # Clean up any temporary files created by the test
        if os.path.exists(output_dir):
            print(f"Cleaning up output directory: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)
            print(f"DISK SPACE AFTER TEST: {get_disk_usage_info()}")
            logger.info(f"DISK SPACE AFTER TEST: {get_disk_usage_info()}")
