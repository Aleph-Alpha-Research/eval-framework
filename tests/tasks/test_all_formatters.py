from typing import Type

import pytest

from eval_framework.task_names import TaskName
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Qwen3ReasoningFormatter
from tests.utils import DatasetPatcher, assert_hash_string

# Tasks to skip temporarily (due to known issues).
SKIP_TASKS = {
    "SQUAD",  # Feature type 'List' not found - datasets library compatibility issue
    "SQUAD2",  # Feature type 'List' not found - datasets library compatibility issue
    "Flores200",  # Could not instantiate: 'utf-8' codec can't decode byte 0x80 in position 108: invalid start byte
}

# Special initialization arguments for specific tasks (can be extended).
SPECIAL_ARGS = {
    "ARC": {"num_fewshot": 1},  # Keep existing 1-shot
    "ARC_DE": {"num_fewshot": 1},
    "ARC_EU20_DE": {"num_fewshot": 1},
    "ARC_EU20_FR": {"num_fewshot": 1},
    "ARC_FI": {"num_fewshot": 1},
    "BigCodeBench": {"num_fewshot": 1},
    "BigCodeBenchInstruct": {"num_fewshot": 1},
    "BigCodeBenchHard": {"num_fewshot": 1},
    "BigCodeBenchHardInstruct": {"num_fewshot": 1},
    "CASEHOLD": {"num_fewshot": 1},
    "ChemBenchMultipleChoice": {"num_fewshot": 1},
    "COPA": {"num_fewshot": 1},
    "DUC_ABSTRACTIVE": {"num_fewshot": 1},
    "DUC_EXTRACTIVE": {"num_fewshot": 1},
    "Flores200": {"num_fewshot": 1},
    "GPQA": {"num_fewshot": 1},
    "GPQA_COT": {"num_fewshot": 1},
    "GSM8K": {"num_fewshot": 1},
    "GSM8KLlamaVersion": {"num_fewshot": 1},
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
    "InfiniteBench_RetrievePassKey1": {"num_fewshot": 0},  # was INFINITE_BENCH_RETRIEVE_PASSKEY1
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
    "MUSR": {"num_fewshot": 0},  ## Trial fix
    "OPENBOOKQA": {"num_fewshot": 1},
    "PAWSX": {"num_fewshot": 2},
    "RenderableStructEval": {"num_fewshot": 0},
    "SCIQ": {"num_fewshot": 1},
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


@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, Qwen3ReasoningFormatter])
@pytest.mark.parametrize("task_name", list(TaskName))
def test_all_tasks_formatter(task_name: TaskName, formatter_cls: Type["BaseFormatter"]) -> None:
    """
    Test that the formatted sample for each (Task, Formatter) pair is consistent by hashing the output.

    Args:
        task_name (TaskName): The task enum value, to be tested.
        formatter_cls (Type[BaseFormatter]): The formatter class to be tested.

    Raises:
        AssertionError: If the hash of the formatter output does not match expectation.
    """
    task_class = task_name.value
    if task_class.__name__ in SKIP_TASKS:
        pytest.skip(f"Skipping {task_class.__name__} appearing in the SKIP_TASKS list.")

    # instantiate the class with the SPECIAL_ARGS dictionary or 1-shot example and fallback to 0-shot if this fails
    try:
        args = SPECIAL_ARGS.get(task_class.__name__, {"num_fewshot": 1})
        with DatasetPatcher(task_class, num_samples=2) as task_instance:
            # Apply any special args to the patched task
            for key, value in args.items():
                setattr(task_instance, key, value)
    except Exception as e:
        print(f"Failed to instantiate task {(task_class.__name__,)}: {e}; retrying with 0-shot")
        try:
            with DatasetPatcher(task_class, num_samples=2, num_fewshot=0) as task_instance:
                pass
        except Exception as e:
            pytest.fail(f"Could not instantiate {task_class.__name__}: {e}")

    formatter = formatter_cls()
    try:
        sample = next(iter(task_instance.iterate_samples(1)))
    except Exception as e:
        pytest.fail(f"No samples for {task_class.__name__}: {e}")

    formatted_sample = formatter.format(sample.messages, output_mode="string")

    possible_completions = sample.possible_completions
    ground_truth = sample.ground_truth

    if possible_completions:
        possible_completions_str: str = "\n".join(f'- "{item}"' for item in possible_completions)
    else:
        possible_completions_str = "None"

    if ground_truth:
        if isinstance(ground_truth, list):
            ground_truth = "\n".join(f'- "{item}"' for item in ground_truth)
        else:
            ground_truth = f'- "{ground_truth}"'
    else:
        ground_truth = "None"

    formatted_sample_with_completions = (
        f"{formatted_sample}\n\nPossible completion:\n{possible_completions_str}\n\nGround truth:\n{ground_truth}"
    )

    assert_hash_string(
        task_name=task_class.__name__,
        suffix_key=formatter_cls.__name__,
        tested_string=formatted_sample_with_completions,
    )
