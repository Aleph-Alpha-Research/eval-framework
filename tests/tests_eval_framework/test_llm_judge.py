from pathlib import Path

import pytest
from tests_eval_framework.utils import _almost_equal

from eval_framework.llm.base import BaseLLM
from eval_framework.main import main
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.registry import get_task

NUM_SAMPLES = 1

# NOTE: Run this tests to make sure redis has the cache in CI

experiment_setups: list[tuple] = []


@pytest.mark.external_api
@pytest.mark.gpu
@pytest.mark.parametrize(
    "test_llms, task_name, expected_results, num_samples, llm_judge_class",
    experiment_setups,
    indirect=["test_llms"],
)
# Test is inherently flaky due to LLM judge non-determinism but an activated redis cache makes it deterministic.
def test_llm_judge_tasks(
    tmp_path: Path,
    test_llms: BaseLLM,
    task_name: str,
    expected_results: dict[str, float],
    num_samples: int,
    llm_judge_class: type[BaseLLM],
) -> None:
    output_dir = tmp_path / "eval"

    task = get_task(task_name)
    task_subjects = getattr(task, "SUBJECTS", None)
    subjects_subset = task_subjects[:3] if task_subjects else None  # limit number of subjects to three

    eval_config = EvalConfig(
        task_name=task_name,
        num_samples=num_samples,
        output_dir=output_dir,
        llm_class=test_llms.__class__,
        llm_judge_class=llm_judge_class,
        save_intermediate_results=False,
        task_subjects=subjects_subset,
    )

    results = main(test_llms, eval_config)

    full_metric_names = [
        (result.metric_name, result.key) for result in results if result.metric_name in list(expected_results.keys())
    ]
    for metric_name, key in set(full_metric_names):
        values = [
            result.value
            for result in results
            if result.metric_name == metric_name and result.key == key and result.value is not None
        ]
        metric_mean = 0 if len(values) == 0 else sum(values) / len(values)
        if key is not None:
            metric_name += f"/{key}"
        assert _almost_equal(expected_results[metric_name], metric_mean), (
            f"Expected {metric_name} to be {expected_results[metric_name]} but got {metric_mean}"
        )
