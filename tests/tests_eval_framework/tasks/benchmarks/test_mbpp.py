import pytest

from eval_framework.tasks.benchmarks.mbpp import MBPPBPB
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.tasks.benchmarks.utils import (
    ExpectedPrompt,
    assert_offline_oneshot_prompt,
    assert_offline_zeroshot_prompt,
    get_task_names_for_module,
    run_formatter_hash_test,
)

_NUM_FEWSHOT = {"MBPP_OLMES": 3}


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", get_task_names_for_module("mbpp"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, num_fewshot=_NUM_FEWSHOT.get(task_name, 1))


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "full"

# Fictional rows following the MBPP format. NOT real examples from the MBPP dataset.
_EVAL_ROW: dict = {
    "text": "Return the number two.",
    "code": "def two():\n    return 2",
    "test_list": ["assert two() == 2", "assert two() != 0"],
}

_FEWSHOT_ROW: dict = {
    "text": "Return the number one.",
    "code": "def one():\n    return 1",
    "test_list": ["assert one() == 1"],
}

# Expected prompts for MBPPBPB. This is a LOGLIKELIHOODS task that scores bits-per-byte on the
# gold reference code: the instruction lists all tests, the assistant cue opens a ```python block,
# and ground truth / the single completion are the reference code (with a leading space).
_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=(
                "You are an expert Python programmer, and here is your task: Return the number two."
                " Your code should pass these tests:\n\nassert two() == 2\nassert two() != 0\n"
            ),
        ),
        Message(role=Role.ASSISTANT, content="```python"),
    ],
    concat="""\
You are an expert Python programmer, and here is your task: Return the number two. Your code should pass these tests:

assert two() == 2
assert two() != 0
```python""",
    ground_truth=" def two():\n    return 2",
    completions=[" def two():\n    return 2"],
)

_ONESHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=(
                "You are an expert Python programmer, and here is your task: Return the number one."
                " Your code should pass these tests:\n\nassert one() == 1\n"
            ),
        ),
        Message(role=Role.ASSISTANT, content="```python\ndef one():\n    return 1\n```"),
        Message(
            role=Role.USER,
            content=(
                "You are an expert Python programmer, and here is your task: Return the number two."
                " Your code should pass these tests:\n\nassert two() == 2\nassert two() != 0\n"
            ),
        ),
        Message(role=Role.ASSISTANT, content="```python"),
    ],
    concat="""\
You are an expert Python programmer, and here is your task: Return the number one. Your code should pass these tests:

assert one() == 1
```python
def one():
    return 1
```

You are an expert Python programmer, and here is your task: Return the number two. Your code should pass these tests:

assert two() == 2
assert two() != 0
```python""",
    ground_truth=_ZEROSHOT.ground_truth,
    completions=_ZEROSHOT.completions,
)


def test_mbppbpb_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        MBPPBPB,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        MBPPBPB,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_ONESHOT,
    )
