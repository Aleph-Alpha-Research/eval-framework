"""Tests for the German HumanEval (EllaMind) tasks.

Tests:
- formatter hash test for every HumanEval variant
- offline prompt assembly tests
"""

from typing import Any

import pytest

import eval_framework.tasks.benchmarks.humaneval_ellamind as humaneval_ellamind
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_humaneval_ellamind_tasks
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    NoStripConcatFormatter,
    Role,
)
from tests.tests_eval_framework.tasks.benchmarks.utils import (
    ExpectedPrompt,
    assert_offline_oneshot_prompt,
    assert_offline_zeroshot_prompt,
    run_formatter_hash_test,
)

# Registry for this test suite only holding humaneval_ellamind tasks
_humaneval_ellamind_registry = Registry()
register_humaneval_ellamind_tasks(registry=_humaneval_ellamind_registry)

# ---------------------------------------------------------------------------
# Formatter hash tests (Hugging Face)
# ---------------------------------------------------------------------------


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _humaneval_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_humaneval_ellamind_registry)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "deu"
_INDENT = "    "

# Fictional rows following the HumanEval format. NOT real examples from the dataset.
# The prompt column sometimes starts with one, or two newlines here, similar to the real dataset.
_EVAL_ROW: dict[str, Any] = {
    "task_id": "HumanEval/0",
    "prompt": f'\n\ndef add(a: int, b: int) -> int:\n{_INDENT}"""Addiert zwei Zahlen."""\n',
    "canonical_solution": f"{_INDENT}return a + b",
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    "entry_point": "add",
}

_FEWSHOT_ROW: dict[str, Any] = {
    "task_id": "HumanEval/1",
    "prompt": f'\ndef square(x: int) -> int:\n{_INDENT}"""Gibt das Quadrat zurück."""\n',
    "canonical_solution": f"{_INDENT}return x * x",
    "test": "def check(candidate):\n    assert candidate(3) == 9\n",
    "entry_point": "square",
}

# Expected prompts (messages, flat concat, ground truth, completions).
# --- HumanEvalDE_OLMES ---
_OLMES_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f'```python\n\n\ndef add(a: int, b: int) -> int:\n{_INDENT}"""Addiert zwei Zahlen."""\n',
        ),
    ],
    concat=f"""\
```python


def add(a: int, b: int) -> int:
{_INDENT}\"\"\"Addiert zwei Zahlen.\"\"\"""",
    ground_truth="Success",
    completions=None,
)

_OLMES_ZEROSHOT_V2 = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f'```python\n\n\ndef add(a: int, b: int) -> int:\n{_INDENT}"""Addiert zwei Zahlen."""\n',
        ),
    ],
    concat=f"""\
```python


def add(a: int, b: int) -> int:
{_INDENT}\"\"\"Addiert zwei Zahlen.\"\"\"""",
    ground_truth="Success",
    completions=None,
)

_OLMES_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f'```python\n\ndef square(x: int) -> int:\n{_INDENT}"""Gibt das Quadrat zurück."""\n',
        ),
        Message(role=Role.ASSISTANT, content=f"{_INDENT}return x * x```"),
        Message(
            role=Role.USER,
            content=f'```python\n\n\ndef add(a: int, b: int) -> int:\n{_INDENT}"""Addiert zwei Zahlen."""\n',
        ),
    ],
    concat=f"""\
```python

def square(x: int) -> int:
{_INDENT}\"\"\"Gibt das Quadrat zurück.\"\"\"
{_INDENT}return x * x```

```python


def add(a: int, b: int) -> int:
{_INDENT}\"\"\"Addiert zwei Zahlen.\"\"\"""",
    ground_truth=_OLMES_ZEROSHOT.ground_truth,
    completions=_OLMES_ZEROSHOT.completions,
)

_OLMES_FEWSHOT_V2 = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f'```python\n\ndef square(x: int) -> int:\n{_INDENT}"""Gibt das Quadrat zurück."""\n',
        ),
        Message(role=Role.ASSISTANT, content=f"{_INDENT}return x * x\n```"),
        Message(
            role=Role.USER,
            content=f'```python\n\n\ndef add(a: int, b: int) -> int:\n{_INDENT}"""Addiert zwei Zahlen."""\n',
        ),
    ],
    concat=f"""\
```python

def square(x: int) -> int:
{_INDENT}\"\"\"Gibt das Quadrat zurück.\"\"\"
{_INDENT}return x * x
```

```python


def add(a: int, b: int) -> int:
{_INDENT}\"\"\"Addiert zwei Zahlen.\"\"\"""",
    ground_truth=_OLMES_ZEROSHOT_V2.ground_truth,
    completions=_OLMES_ZEROSHOT_V2.completions,
)

# --- HumanEvalDE_BPB_OLMES_V2 ---
# The BPB_V2 variant mirrors the OLMES completion prompt exactly (including the ```python fences),
# so that the loglikelihood-scored prompt and the generation prompt are identical.
_BPB_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f'```python\n\n\ndef add(a: int, b: int) -> int:\n{_INDENT}"""Addiert zwei Zahlen."""\n',
        ),
    ],
    concat=f"""\
```python


def add(a: int, b: int) -> int:
{_INDENT}\"\"\"Addiert zwei Zahlen.\"\"\"""",
    ground_truth=f"{_INDENT}return a + b\n```",
    completions=[f"{_INDENT}return a + b\n```"],
)

_BPB_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f'```python\n\ndef square(x: int) -> int:\n{_INDENT}"""Gibt das Quadrat zurück."""\n',
        ),
        Message(role=Role.ASSISTANT, content=f"{_INDENT}return x * x\n```"),
        Message(
            role=Role.USER,
            content=f'```python\n\n\ndef add(a: int, b: int) -> int:\n{_INDENT}"""Addiert zwei Zahlen."""\n',
        ),
    ],
    concat=f"""\
```python

def square(x: int) -> int:
{_INDENT}\"\"\"Gibt das Quadrat zurück.\"\"\"
{_INDENT}return x * x
```

```python


def add(a: int, b: int) -> int:
{_INDENT}\"\"\"Addiert zwei Zahlen.\"\"\"""",
    ground_truth=_BPB_ZEROSHOT.ground_truth,
    completions=_BPB_ZEROSHOT.completions,
)


# --- TESTS ---
def test_humanevalde_olmes_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        humaneval_ellamind.HumanEvalDE_OLMES,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_OLMES_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        humaneval_ellamind.HumanEvalDE_OLMES,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_OLMES_FEWSHOT,
    )


def test_HumanEvalDE_OLMES_V2_olmes_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        humaneval_ellamind.HumanEvalDE_OLMES_V2,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_OLMES_ZEROSHOT_V2,
    )
    assert_offline_oneshot_prompt(
        humaneval_ellamind.HumanEvalDE_OLMES_V2,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_OLMES_FEWSHOT_V2,
    )


def test_humanevalde_olmes_bpb_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        humaneval_ellamind.HumanEvalDE_BPB_OLMES_V2,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        humaneval_ellamind.HumanEvalDE_BPB_OLMES_V2,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )


@pytest.mark.slow
def test_humaneval_completion_bpb_same():
    task = humaneval_ellamind.HumanEvalDE_BPB_OLMES_V2.with_overwrite(
        num_fewshot=3, custom_subjects=None, custom_hf_revision=None
    )
    for task1_sample in task.iterate_samples(1):
        break

    humaneval = ConcatFormatter().format(task1_sample.messages)
    task_compl = humaneval_ellamind.HumanEvalDE_OLMES.with_overwrite(
        num_fewshot=3, custom_subjects=None, custom_hf_revision=None
    )
    for task2_sample in task_compl.iterate_samples(1):
        break

    humaneval_compl = ConcatFormatter().format(task2_sample.messages)
    assert humaneval == humaneval_compl
    assert task1_sample.messages == task2_sample.messages
