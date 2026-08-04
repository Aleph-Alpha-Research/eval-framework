"""Tests for the German MBPP (EllaMind) tasks.

Tests:
- formatter hash test for every MBPP variant
- offline prompt assembly tests for the generative OLMES format and the BPB format
"""

from typing import Any
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

import eval_framework.tasks.benchmarks.mbpp_ellamind as mbpp_ellamind
from eval_framework.tasks.base import Sample
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_mbpp_ellamind_tasks
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    Role,
)
from tests.tests_eval_framework.tasks.benchmarks.utils import (
    ExpectedPrompt,
    assert_offline_oneshot_prompt,
    assert_offline_zeroshot_prompt,
    run_formatter_hash_test,
)

# Registry for this test suite only holding mbpp_ellamind tasks
_mbpp_ellamind_registry = Registry()
register_mbpp_ellamind_tasks(registry=_mbpp_ellamind_registry)

# ---------------------------------------------------------------------------
# Formatter hash tests (Hugging Face)
# ---------------------------------------------------------------------------


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _mbpp_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_mbpp_ellamind_registry)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "deu"
_INDENT = "    "

# Fictional rows following the MBPP format. NOT real examples from the MBPP dataset.
# The ``code`` fields intentionally use ``\r\n`` to mirror the real dataset, whose
# ``code`` values ship Windows-style carriage returns.
_EVAL_ROW: dict[str, Any] = {
    "text": "Schreibe eine Funktion, die zwei Zahlen addiert.",
    "code": f"def add(a, b):\r\n{_INDENT}return a + b",
    "test_list": ["assert add(1, 2) == 3"],
}

_FEWSHOT_ROW: dict[str, Any] = {
    "text": "Schreibe eine Funktion, die eine Zahl quadriert.",
    "code": f"def square(x):\r\n{_INDENT}return x * x",
    "test_list": ["assert square(3) == 9"],
}

# --- MBPPDE_OLMES (generative; German instruction wrapper, test execution) ---

_INSTRUCTION_PREFIX = (
    "Bitte erstelle ein in sich geschlossenes Python-Skript, "
    "das das folgende Problem in einem Markdown-Code-Block löst:"
)
_CUE = "Hier ist die fertige Funktion:\n\n```python\n"

# Expected prompts (messages, flat concat, ground truth, completions).
# --- MBPPDE_OLMES ---
_OLMES_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f"{_INSTRUCTION_PREFIX}\n```\nSchreibe eine Funktion, die zwei Zahlen addiert.\nassert add(1, 2) == 3\n```\n",
        ),
        Message(role=Role.ASSISTANT, content=_CUE),
    ],
    concat="""\
Bitte erstelle ein in sich geschlossenes Python-Skript, das das folgende Problem in einem Markdown-Code-Block löst:
```
Schreibe eine Funktion, die zwei Zahlen addiert.
assert add(1, 2) == 3
```
Hier ist die fertige Funktion:

```python""",
    ground_truth="['assert add(1, 2) == 3']",
    completions=None,
)

_OLMES_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f"{_INSTRUCTION_PREFIX}\n```\nSchreibe eine Funktion, die eine Zahl quadriert.\nassert square(3) == 9\n```\n",
        ),
        Message(role=Role.ASSISTANT, content="def square(x):\r\n    return x * x\n"),
        Message(
            role=Role.USER,
            content=f"{_INSTRUCTION_PREFIX}\n```\nSchreibe eine Funktion, die zwei Zahlen addiert.\nassert add(1, 2) == 3\n```\n",
        ),
        Message(role=Role.ASSISTANT, content=_CUE),
    ],
    # Note: the ``code`` field contains Windows-style carriage returns (``\r\n``), which in the generative variant are currently not normalized.
    # As a result, we get the additional ``\\r`` in the one shot example.
    concat="""\
Bitte erstelle ein in sich geschlossenes Python-Skript, das das folgende Problem in einem Markdown-Code-Block löst:
```
Schreibe eine Funktion, die eine Zahl quadriert.
assert square(3) == 9
```
def square(x):\r
    return x * x


Bitte erstelle ein in sich geschlossenes Python-Skript, das das folgende Problem in einem Markdown-Code-Block löst:
```
Schreibe eine Funktion, die zwei Zahlen addiert.
assert add(1, 2) == 3
```
Hier ist die fertige Funktion:

```python""",
    ground_truth=_OLMES_ZEROSHOT.ground_truth,
    completions=_OLMES_ZEROSHOT.completions,
)

# --- MBPPDE_BPB_OLMES ---
_BPB_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Schreibe eine Funktion, die zwei Zahlen addiert.\n```python\n"),
    ],
    concat="""\
Schreibe eine Funktion, die zwei Zahlen addiert.
```python""",
    ground_truth=f"def add(a, b):\n{_INDENT}return a + b\n```",
    completions=[f"def add(a, b):\n{_INDENT}return a + b\n```"],
)

_BPB_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Schreibe eine Funktion, die eine Zahl quadriert.\n```python\n"),
        Message(role=Role.ASSISTANT, content=f"def square(x):\n{_INDENT}return x * x\n```"),
        Message(role=Role.USER, content="Schreibe eine Funktion, die zwei Zahlen addiert.\n```python\n"),
    ],
    concat="""\
Schreibe eine Funktion, die eine Zahl quadriert.
```python
def square(x):
    return x * x
```

Schreibe eine Funktion, die zwei Zahlen addiert.
```python""",
    ground_truth=_BPB_ZEROSHOT.ground_truth,
    completions=_BPB_ZEROSHOT.completions,
)


# --- TESTS ---
def test_mbppde_olmes_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        mbpp_ellamind.MBPPDE_OLMES,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_OLMES_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        mbpp_ellamind.MBPPDE_OLMES,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_OLMES_FEWSHOT,
    )


def test_mbppde_bpb_olmes_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        mbpp_ellamind.MBPPDE_BPB_OLMES,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        mbpp_ellamind.MBPPDE_BPB_OLMES,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )


# ---------------------------------------------------------------------------
# MBPPDE_EvalPlus / MBPPDE_BPB_EvalPlus (EvalPlus prompt format, German pendants)
# ---------------------------------------------------------------------------

# Fictional rows following the MBPP format. NOT real examples from the MBPP dataset.
# The German EvalPlus tasks sample few-shots from the same split; we patch
# ``_sample_fewshot_examples`` to inject these short fakes instead.
_EP_FEWSHOT_EXAMPLES: list[dict] = [
    {"text": "Gib die Zahl eins zurück.", "code": "def eins():\n    return 1", "test_list": ["assert eins() == 1"]},
]

_EP_EVAL_ROW: dict = {
    "text": "Gib die Zahl zwei zurück.",
    "code": "def zwei():\n    return 2",
    "test_list": ["assert zwei() == 2", "assert zwei() != 0"],
}

# --- MBPPDE_EvalPlus (generative; test execution) ---
_EVALPLUS_EXPECTED = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=(
                "Bitte erstelle ein in sich geschlossenes Python-Skript, das das folgende Problem"
                " in einem Markdown-Code-Block löst:\n```\nGib die Zahl eins zurück.\nassert eins() == 1\n```\n"
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "Hier ist ein Python-Skript mit einer in sich geschlossenen Funktion, die das Problem"
                " löst und die entsprechenden Tests besteht:\n```python\ndef eins():\n    return 1\n```"
            ),
        ),
        Message(
            role=Role.USER,
            content=(
                "Bitte erstelle ein in sich geschlossenes Python-Skript, das das folgende Problem"
                " in einem Markdown-Code-Block löst:\n```\nGib die Zahl zwei zurück.\nassert zwei() == 2\n```\n"
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "Hier ist ein Python-Skript mit einer in sich geschlossenen Funktion, die das Problem"
                " löst und die entsprechenden Tests besteht:\n```python"
            ),
        ),
    ],
    concat="""\
Bitte erstelle ein in sich geschlossenes Python-Skript, das das folgende Problem in einem Markdown-Code-Block löst:
```
Gib die Zahl eins zurück.
assert eins() == 1
```
Hier ist ein Python-Skript mit einer in sich geschlossenen Funktion, die das Problem löst und die entsprechenden Tests besteht:
```python
def eins():
    return 1
```

Bitte erstelle ein in sich geschlossenes Python-Skript, das das folgende Problem in einem Markdown-Code-Block löst:
```
Gib die Zahl zwei zurück.
assert zwei() == 2
```
Hier ist ein Python-Skript mit einer in sich geschlossenen Funktion, die das Problem löst und die entsprechenden Tests besteht:
```python""",
    ground_truth="['assert zwei() == 2', 'assert zwei() != 0']",
    completions=None,
)

# --- MBPPDE_BPB_EvalPlus (loglikelihood; identical prompt, BPB scoring of the reference code) ---
_BPB_EVALPLUS_EXPECTED = ExpectedPrompt(
    messages=_EVALPLUS_EXPECTED.messages,
    concat=_EVALPLUS_EXPECTED.concat,
    ground_truth="\ndef zwei():\n    return 2\n```",
    completions=["\ndef zwei():\n    return 2\n```"],
)


def _assert_sample_matches(sample: Sample, expected: ExpectedPrompt) -> None:
    assert sample.messages == expected.messages
    assert ConcatFormatter().format(sample.messages, output_mode="string") == expected.concat
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.completions


def test_mbppde_evalplus_offline_prompt_formatting() -> None:
    def mock_fewshot_examples(self: Any, item: dict[str, Any]) -> list[dict]:
        return list(_EP_FEWSHOT_EXAMPLES)

    task = mbpp_ellamind.MBPPDE_EvalPlus.with_overwrite(
        num_fewshot=3, custom_subjects=[_SUBJECT], custom_hf_revision=None
    )
    mock_dataset = DatasetDict({task.SAMPLE_SPLIT: Dataset.from_list([_EP_EVAL_ROW])})

    with patch.object(mbpp_ellamind.MBPPDE_EvalPlus, "_sample_fewshot_examples", mock_fewshot_examples):
        with patch.object(task, "_load_hf_dataset", return_value=mock_dataset):
            sample = next(iter(task.iterate_samples(1)))

    _assert_sample_matches(sample, _EVALPLUS_EXPECTED)


def test_mbppde_bpb_evalplus_offline_prompt_formatting() -> None:
    def mock_fewshot_examples(self: Any, item: dict[str, Any]) -> list[dict]:
        return list(_EP_FEWSHOT_EXAMPLES)

    task = mbpp_ellamind.MBPPDE_BPB_EvalPlus.with_overwrite(
        num_fewshot=3, custom_subjects=[_SUBJECT], custom_hf_revision=None
    )
    mock_dataset = DatasetDict({task.SAMPLE_SPLIT: Dataset.from_list([_EP_EVAL_ROW])})

    with patch.object(mbpp_ellamind.MBPPDE_BPB_EvalPlus, "_sample_fewshot_examples", mock_fewshot_examples):
        with patch.object(task, "_load_hf_dataset", return_value=mock_dataset):
            sample = next(iter(task.iterate_samples(1)))

    _assert_sample_matches(sample, _BPB_EVALPLUS_EXPECTED)
