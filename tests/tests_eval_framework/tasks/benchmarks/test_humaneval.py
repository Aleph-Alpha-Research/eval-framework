from typing import Any

import pytest

from eval_framework.tasks.benchmarks.humaneval import HumanEval, HumanEval_OLMES, HumanEvalBPB_V2, HumanEvalInstruct
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_humaneval_tasks
from eval_framework.tasks.utils import run_python_code
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.tasks.benchmarks.utils import (
    ExpectedPrompt,
    assert_offline_oneshot_prompt,
    assert_offline_zeroshot_prompt,
    run_formatter_hash_test,
)
from tests.tests_eval_framework.utils import DatasetPatcher

_NUM_FEWSHOT = {"HumanEval_OLMES": 3}


class TestHumanEvalCode:
    @pytest.fixture
    def human_eval_task(self) -> HumanEval:
        with DatasetPatcher(HumanEval, num_fewshot=0) as patched_task:
            return patched_task

    def test_code_is_executed(self, human_eval_task: HumanEval) -> None:
        assert len(human_eval_task.SUBJECTS) > 0
        human_eval_task._load_dataset(human_eval_task.SUBJECTS[0])
        i = 0
        for i, item in enumerate(human_eval_task.dataset[human_eval_task.SAMPLE_SPLIT][:10]):
            sample = human_eval_task._create_samples(item, i, human_eval_task.SUBJECTS[0])[0]
            formatted_code = human_eval_task.post_process_generated_completion(item["canonical_solution"], sample)
            assert run_python_code(formatted_code).endswith("True")
            formatted_code = human_eval_task.post_process_generated_completion("", sample)
            assert not run_python_code(formatted_code).endswith("True")
        assert i == 9


class TestHumanEvalOLMES:
    @pytest.fixture
    def human_eval_olmes_task(self) -> HumanEval_OLMES:
        with DatasetPatcher(HumanEval_OLMES, num_fewshot=3) as patched_task:
            return patched_task

    def test_code_is_executed(self, human_eval_olmes_task: HumanEval_OLMES) -> None:
        assert len(human_eval_olmes_task.SUBJECTS) > 0
        subject = human_eval_olmes_task.SUBJECTS[0]
        human_eval_olmes_task._load_dataset(subject)
        i = 0
        for i, item in enumerate(human_eval_olmes_task.dataset[human_eval_olmes_task.SAMPLE_SPLIT][:10]):
            item["subject"] = subject
            sample = human_eval_olmes_task._create_samples(item, i, subject)[0]
            formatted_code = human_eval_olmes_task.post_process_generated_completion(item["canonical_solution"], sample)
            assert run_python_code(formatted_code).endswith("True")
            formatted_code = human_eval_olmes_task.post_process_generated_completion("", sample)
            assert not run_python_code(formatted_code).endswith("True")
        assert i == 9

    def test_olmes_settings(self, human_eval_olmes_task: HumanEval_OLMES) -> None:
        assert human_eval_olmes_task.num_fewshot == 3
        assert human_eval_olmes_task.max_tokens == 1024
        assert "\nclass" in human_eval_olmes_task.stop_sequences
        assert "\nif" in human_eval_olmes_task.stop_sequences
        assert "\nprint" in human_eval_olmes_task.stop_sequences
        assert "\n#" in human_eval_olmes_task.stop_sequences
        assert "\n```" in human_eval_olmes_task.stop_sequences
        assert human_eval_olmes_task.SAMPLE_SPLIT == "test"
        assert human_eval_olmes_task.FEWSHOT_SPLIT == "test"

    def test_olmes_prompt_format(self, human_eval_olmes_task: HumanEval_OLMES) -> None:
        human_eval_olmes_task._load_dataset(human_eval_olmes_task.SUBJECTS[0])
        item = human_eval_olmes_task.dataset[human_eval_olmes_task.SAMPLE_SPLIT][0]
        instruction = human_eval_olmes_task._get_instruction_text(item)
        assert instruction.startswith("```python\n")
        assert instruction == "```python\n" + item["prompt"]

        fewshot_target = human_eval_olmes_task._get_fewshot_target_text(item)
        assert fewshot_target.endswith("```")
        assert fewshot_target == item["canonical_solution"] + "```"


class TestHumanEvalInstructCode:
    @pytest.fixture
    def human_eval_task_inst(self) -> HumanEvalInstruct:
        with DatasetPatcher(HumanEvalInstruct, num_fewshot=0) as patched_task:
            return patched_task

    def test_code_is_executed(self, human_eval_task_inst: HumanEvalInstruct) -> None:
        assert len(human_eval_task_inst.SUBJECTS) > 0
        human_eval_task_inst._load_dataset(human_eval_task_inst.SUBJECTS[0])
        i = 0
        for i, item in enumerate(human_eval_task_inst.dataset[human_eval_task_inst.SAMPLE_SPLIT][:10]):
            sample = human_eval_task_inst._create_samples(item, i, human_eval_task_inst.SUBJECTS[0])[0]
            completion = item["canonical_solution"]
            formatted_code = human_eval_task_inst.post_process_generated_completion(completion, sample)
            assert run_python_code(formatted_code).endswith("True")
        assert i == 9


# Registry for this test suite only holding humaneval tasks
_humaneval_registry = Registry()
register_humaneval_tasks(registry=_humaneval_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _humaneval_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(
        task_name, formatter_cls, num_fewshot=_NUM_FEWSHOT.get(task_name, 1), registry=_humaneval_registry
    )


_INDENT = " " * 4

# Fictional rows following the HumanEval format. NOT real examples from the dataset.
# The prompt column sometimes starts with one, or two newlines here, similar to the real dataset.
_EVAL_ROW: dict[str, Any] = {
    "task_id": "HumanEval/0",
    "prompt": f'\n\ndef add(a: int, b: int) -> int:\n{_INDENT}"""Adds two numbers."""\n',
    "canonical_solution": f"{_INDENT}return a + b",
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    "entry_point": "add",
}

_FEWSHOT_ROW: dict[str, Any] = {
    "task_id": "HumanEval/1",
    "prompt": f'\ndef square(x: int) -> int:\n{_INDENT}"""Returns the square."""\n',
    "canonical_solution": f"{_INDENT}return x * x",
    "test": "def check(candidate):\n    assert candidate(3) == 9\n",
    "entry_point": "square",
}

# --- HumanEvalBPB_V2 ---
_BPB_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f'```python\n\n\ndef add(a: int, b: int) -> int:\n{_INDENT}"""Adds two numbers."""\n',
        ),
    ],
    concat=f"""\
```python


def add(a: int, b: int) -> int:
{_INDENT}\"\"\"Adds two numbers.\"\"\"""",
    ground_truth=f"{_INDENT}return a + b",
    completions=[f"{_INDENT}return a + b"],
)

_BPB_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f'```python\n\ndef square(x: int) -> int:\n{_INDENT}"""Returns the square."""\n',
        ),
        Message(role=Role.ASSISTANT, content=f"{_INDENT}return x * x\n```"),
        Message(
            role=Role.USER,
            content=f'```python\n\n\ndef add(a: int, b: int) -> int:\n{_INDENT}"""Adds two numbers."""\n',
        ),
    ],
    concat=f"""\
```python

def square(x: int) -> int:
{_INDENT}\"\"\"Returns the square.\"\"\"
{_INDENT}return x * x
```

```python


def add(a: int, b: int) -> int:
{_INDENT}\"\"\"Adds two numbers.\"\"\"""",
    ground_truth=_BPB_ZEROSHOT.ground_truth,
    completions=_BPB_ZEROSHOT.completions,
)


def test_humaneval_bpb_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        HumanEvalBPB_V2,
        eval_row=_EVAL_ROW,
        subjects=[HumanEvalBPB_V2.SUBJECTS[0]],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        HumanEvalBPB_V2,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[HumanEvalBPB_V2.SUBJECTS[0]],
        expected=_BPB_FEWSHOT,
    )


@pytest.mark.slow
def test_humaneval_completion_bpb_same():
    task = HumanEvalBPB_V2.with_overwrite(num_fewshot=3, custom_subjects=None, custom_hf_revision=None)
    for task1_sample in task.iterate_samples(1):
        break

    humaneval = ConcatFormatter().format(task1_sample.messages)
    task_compl = HumanEval_OLMES.with_overwrite(num_fewshot=3, custom_subjects=None, custom_hf_revision=None)
    for task2_sample in task_compl.iterate_samples(1):
        break

    humaneval_compl = ConcatFormatter().format(task2_sample.messages)
    assert humaneval == humaneval_compl
    assert task1_sample.messages == task2_sample.messages
