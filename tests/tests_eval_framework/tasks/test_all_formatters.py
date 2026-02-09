"""
Formatter hash tests for all registered tasks.

Validates that formatters produce consistent output by comparing MD5 hashes.
Uses full HuggingFace datasets with random seed 42 for deterministic behavior.

Usage:
    pytest -m formatter_hash
"""

import random
import sys
from collections.abc import Generator
from typing import Any

import pytest

from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.registry import get_task, registered_task_names
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter
from tests.tests_eval_framework.utils import assert_hash_string

# Special initialization arguments for specific tasks
SPECIAL_ARGS: dict[str, dict[str, Any]] = {
    "ARC": {"num_fewshot": 1},
    "ARC_DE": {"num_fewshot": 1},
    "ARC_EU20_DE": {"num_fewshot": 1},
    "ARC_EU20_FR": {"num_fewshot": 1},
    "ARC_FI": {"num_fewshot": 1},
    "BigCodeBench": {"num_fewshot": 1},
    "BigCodeBenchInstruct": {"num_fewshot": 1},
    "BigCodeBenchHard": {"num_fewshot": 1},
    "BigCodeBenchHardInstruct": {"num_fewshot": 1},
    "CASEHOLD": {"num_fewshot": 1},
    "ChemBench": {"num_fewshot": 1},
    "COPA": {"num_fewshot": 1},
    "DUC_ABSTRACTIVE": {"num_fewshot": 1},
    "DUC_EXTRACTIVE": {"num_fewshot": 1},
    "Flores200": {"num_fewshot": 1},
    "Global-MMLU": {"num_fewshot": 1},
    "GPQA": {"num_fewshot": 1},
    "GPQA_COT": {"num_fewshot": 1},
    "GSM8K": {"num_fewshot": 1},
    "GSM8KEvalHarness": {"num_fewshot": 1},
    "GSM8KReasoning": {"num_fewshot": 0},
    "GSM8K_EU20_DE": {"num_fewshot": 1},
    "GSM8K_EU20_FR": {"num_fewshot": 1},
    "HELLASWAG": {"num_fewshot": 1},
    "HELLASWAG_DE": {"num_fewshot": 1},
    "HELLASWAG_EU20_DE": {"num_fewshot": 1},
    "HELLASWAG_EU20_FR": {"num_fewshot": 1},
    "InfiniteBench_CodeDebug": {"num_fewshot": 0},
    "InfiniteBench_CodeRun": {"num_fewshot": 0},
    "InfiniteBench_EnDia": {"num_fewshot": 0},
    "InfiniteBench_EnMC": {"num_fewshot": 0},
    "InfiniteBench_EnQA": {"num_fewshot": 0},
    "InfiniteBench_MathFind": {"num_fewshot": 0},
    "InfiniteBench_RetrieveKV2": {"num_fewshot": 0},
    "InfiniteBench_RetrieveNumber": {"num_fewshot": 0},
    "InfiniteBench_RetrievePassKey1": {"num_fewshot": 0},
    "MATH": {"num_fewshot": 1},
    "MATHLvl5": {"num_fewshot": 1},
    "MATH500": {"num_fewshot": 1},
    "MBPP": {"num_fewshot": 1},
    "MBPP_SANITIZED": {"num_fewshot": 1},
    "MBPP_PROMPT_WITHOUT_TESTS": {"num_fewshot": 1},
    "MBPP_PROMPT_WITHOUT_TESTS_SANITIZED": {"num_fewshot": 1},
    "MMLU": {"num_fewshot": 1},
    "FullTextMMLU": {"num_fewshot": 1},
    "MMLU_EU20_DE": {"num_fewshot": 1},
    "MMLU_EU20_FR": {"num_fewshot": 1},
    "MMLU_DE": {"num_fewshot": 1},
    "MMLU_PRO": {"num_fewshot": 1},
    "MMLU_PRO_COT": {"num_fewshot": 1},
    "MMLU_COT": {"num_fewshot": 1},
    "MMMLU": {"num_fewshot": 1},
    "MMMLU_GERMAN_COT": {"num_fewshot": 1},
    "OPENBOOKQA": {"num_fewshot": 1},
    "PAWSX": {"num_fewshot": 2},
    "RenderableStructEval": {"num_fewshot": 0},
    "SCIQ": {"num_fewshot": 1},
    "SCIQEvalHarness": {"num_fewshot": 1},
    "SQUAD": {"num_fewshot": 1},
    "SQUAD2": {"num_fewshot": 1},
    "SPHYR": {"num_fewshot": 0},
    "StructEval": {"num_fewshot": 0},
    "TRIVIAQA": {"num_fewshot": 1},
    "TRUTHFULQA": {"num_fewshot": 1},
    "TRUTHFULQA_DE": {"num_fewshot": 1},
    "TRUTHFULQA_EU20_DE": {"num_fewshot": 1},
    "TRUTHFULQA_EU20_FR": {"num_fewshot": 1},
    "TRUTHFULQA_PERTURBED": {"num_fewshot": 1},
    "TRUTHFULQA_PERTURBED_DE": {"num_fewshot": 1},
    "WINOGENDER": {"num_fewshot": 1},
    "WINOGRANDE": {"num_fewshot": 1},
    "WINOX_DE": {"num_fewshot": 1},
    "WINOX_FR": {"num_fewshot": 1},
    "WMT14": {"num_fewshot": 1},
    "WMT16": {"num_fewshot": 1},
    "WMT20": {"num_fewshot": 1},
    "WMT14_INSTRUCT": {"num_fewshot": 1},
    "WMT16_INSTRUCT": {"num_fewshot": 1},
    "WMT20_INSTRUCT": {"num_fewshot": 1},
}

TASKS_TO_TEST = set(registered_task_names())


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment() -> Generator[None, None, None]:
    """
    Configure test environment for deterministic behavior.

    Sets random seeds for reproducible test results.
    """
    import random

    import numpy as np

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Set HuggingFace datasets seed if available
    try:
        import datasets

        datasets.set_random_seed(42)
    except (ImportError, AttributeError):
        pass

    print("\nFormatter tests using full HuggingFace datasets (seed=42)", file=sys.stderr)
    print("", file=sys.stderr)

    yield


def _load_sample(task_class: type[BaseTask], args: dict[str, Any]) -> tuple[Any, str]:
    """
    Load a sample using full HuggingFace datasets.

    Args:
        task_class: The task class to instantiate
        args: Special arguments for the task

    Returns:
        Tuple of (sample, task_class_name)

    Raises:
        Exception: If sample cannot be loaded even with 0-shot fallback
    """
    try:
        num_fewshot = args.get("num_fewshot", 1)
        task_instance = task_class.with_overwrite(
            num_fewshot=num_fewshot,
            custom_subjects=None,
            custom_hf_revision=None,
        )

        original_method = task_instance._sample_fewshot_examples

        def deterministic_fewshot(item: dict) -> list:
            task_instance.rnd = random.Random(42)  # Fresh seed each time
            return original_method(item)

        task_instance._sample_fewshot_examples = deterministic_fewshot

        sample = next(iter(task_instance.iterate_samples(1)))
        return sample, task_class.__name__
    except Exception as e:
        # Fallback to 0-shot
        print(
            f"Failed to instantiate task {task_class.__name__}: {e}; retrying with 0-shot",
            file=sys.stderr,
        )
        task_instance = task_class.with_overwrite(
            num_fewshot=0,
            custom_subjects=None,
            custom_hf_revision=None,
        )
        sample = next(iter(task_instance.iterate_samples(1)))
        return sample, task_class.__name__


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", sorted(TASKS_TO_TEST))
def test_all_tasks_formatter(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    """
    Test that the formatted sample for each (Task, Formatter) pair is consistent.

    This test validates formatter output by computing and comparing MD5 hashes.
    Hash mismatches indicate changes in:
    - Formatter logic
    - Task prompt construction
    - Dataset samples

    Args:
        task_name: The task name to test
        formatter_cls: The formatter class to test (Llama3Formatter or ConcatFormatter)

    Raises:
        AssertionError: If the hash of the formatter output does not match the expected value
    """
    # Skip WMT tasks - sacrebleu file loading has non-determinism
    if "WMT" in task_name:
        pytest.skip(f"Skipping {task_name}: WMT tasks use sacrebleu with non-deterministic file loading")

    task_class = get_task(task_name)
    args = SPECIAL_ARGS.get(task_class.__name__, {"num_fewshot": 1})

    # Load sample
    try:
        sample, task_class_name = _load_sample(task_class, args)
    except Exception as e:
        pytest.fail(f"Could not instantiate {task_class.__name__}: {e}")

    # Format the sample
    formatter = formatter_cls()
    formatted_sample = formatter.format(sample.messages, output_mode="string")

    # Build comparison string with possible completions and ground truth
    possible_completions = sample.possible_completions
    ground_truth = sample.ground_truth

    if possible_completions:
        possible_completions_str: str = "\n".join(f'- "{item}"' for item in possible_completions)
    else:
        possible_completions_str = "None"

    if ground_truth:
        if isinstance(ground_truth, list):
            ground_truth_str = "\n".join(f'- "{item}"' for item in ground_truth)
        else:
            ground_truth_str = f'- "{ground_truth}"'
    else:
        ground_truth_str = "None"

    formatted_sample_with_completions = (
        f"{formatted_sample}\n\nPossible completion:\n{possible_completions_str}\n\nGround truth:\n{ground_truth_str}"
    )

    # Verify hash consistency
    assert_hash_string(
        task_name=task_class_name,
        suffix_key=formatter_cls.__name__,
        tested_string=formatted_sample_with_completions,
    )
