from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from eval_framework.tasks.benchmarks.mbpp import _OLMES_FEWSHOT_EXAMPLES, MBPP_OLMES, MBPP_BPB_EvalPlus, MBPP_EvalPlus
from eval_framework.tasks.utils import run_python_code
from template_formatting.formatter import ConcatFormatter, Message, Role
from tests.tests_eval_framework.tasks.benchmarks.utils import ExpectedPrompt
from tests.tests_eval_framework.utils import DatasetPatcher


class TestMBPP_OLMES:
    @pytest.fixture
    def task(self) -> MBPP_OLMES:
        with DatasetPatcher(MBPP_OLMES, num_fewshot=3, num_samples=10) as patched_task:
            return patched_task

    def test_stop_sequences(self) -> None:
        task = MBPP_OLMES(num_fewshot=3)
        assert task.stop_sequences == ["```", '\n"""', "\nassert", "\n#"]

    def test_instruction_uses_evalplus_format(self, task: MBPP_OLMES) -> None:
        task._load_dataset(task.SUBJECTS[0])
        item = task.dataset[task.SAMPLE_SPLIT][0]
        item["subject"] = task.SUBJECTS[0]
        instruction = task._get_instruction_text(item)

        expected_prefix = (
            "Please provide a self-contained Python script that solves the following problem"
            " in a markdown code block:\n```\n"
        )
        assert instruction.startswith(expected_prefix)
        assert instruction.endswith("\n```\n")
        assert item["test_list"][0] in instruction

    def test_instruction_contains_only_one_test(self, task: MBPP_OLMES) -> None:
        task._load_dataset(task.SUBJECTS[0])
        item = task.dataset[task.SAMPLE_SPLIT][0]
        item["subject"] = task.SUBJECTS[0]
        instruction = task._get_instruction_text(item)

        for test in item["test_list"][1:]:
            assert test not in instruction

    def test_cue_text(self, task: MBPP_OLMES) -> None:
        task._load_dataset(task.SUBJECTS[0])
        item = task.dataset[task.SAMPLE_SPLIT][0]
        cue = task._get_cue_text(item)
        assert cue == "Here is the completed function:\n\n```python\n"

    def test_fewshot_examples_are_hardcoded(self, task: MBPP_OLMES) -> None:
        task._load_dataset(task.SUBJECTS[0])
        item = task.dataset[task.SAMPLE_SPLIT][0]

        examples = task._sample_fewshot_examples(item)
        assert len(examples) == 3
        assert examples[0]["text"] == _OLMES_FEWSHOT_EXAMPLES[0]["text"]
        assert examples[1]["text"] == _OLMES_FEWSHOT_EXAMPLES[1]["text"]
        assert examples[2]["text"] == _OLMES_FEWSHOT_EXAMPLES[2]["text"]

    def test_fewshot_examples_are_deterministic(self, task: MBPP_OLMES) -> None:
        task._load_dataset(task.SUBJECTS[0])
        item = task.dataset[task.SAMPLE_SPLIT][0]

        examples_1 = task._sample_fewshot_examples(item)
        examples_2 = task._sample_fewshot_examples(item)
        assert examples_1 == examples_2

    def test_fewshot_target_is_code_with_newline(self) -> None:
        task = MBPP_OLMES(num_fewshot=3)
        for example in _OLMES_FEWSHOT_EXAMPLES:
            target = task._get_fewshot_target_text(example)
            assert target == example["code"] + "\n"
            assert "```" not in target

    def test_code_execution_with_canonical_solution(self, task: MBPP_OLMES) -> None:
        task._load_dataset(task.SUBJECTS[0])
        for i, item in enumerate(task.dataset[task.SAMPLE_SPLIT][:5]):
            # Verify that canonical code + test asserts execute correctly.
            # We call _code_expander directly because in real usage the LLM
            # engine truncates output at stop sequences (e.g. \n#) before it
            # reaches post_process, and canonical solutions may contain those
            # sequences (comments, asserts) that would corrupt the test.
            code = MBPP_OLMES._code_expander(item["code"] + "\n", str(item["test_list"]))
            result = run_python_code(code)
            assert result.endswith("True"), f"Item {i} failed: {result}"

    def test_prompt_format_matches_oe_eval(self, task: MBPP_OLMES) -> None:
        """Verify the assembled prompt has the expected structure with ConcatFormatter."""
        task._load_dataset(task.SUBJECTS[0])
        item = task.dataset[task.SAMPLE_SPLIT][0]
        item["subject"] = task.SUBJECTS[0]
        sample = task._create_samples(item, 0, task.SUBJECTS[0])[0]

        formatter = ConcatFormatter()
        formatted = formatter.format(sample.messages, output_mode="string")

        assert "Please provide a self-contained Python script" in formatted
        assert "Here is the completed function:" in formatted
        assert "```python" in formatted

        fewshot_count = formatted.count("Please provide a self-contained Python script")
        assert fewshot_count == 4, f"Expected 4 occurrences (3 fewshot + 1 eval), got {fewshot_count}"


# ---------------------------------------------------------------------------
# Offline prompt assembly test (uses fictional dataset + fictional fewshot)
# ---------------------------------------------------------------------------

_SUBJECT = "full"

# Fictional rows following the MBPP format. NOT real examples from the MBPP dataset.
# The fewshot examples are normally hardcoded in the task (``_OLMES_FEWSHOT_EXAMPLES``);
# we patch ``_sample_fewshot_examples`` to inject these short fakes instead.
_FEWSHOT_EXAMPLES: list[dict] = [
    {"text": "Return the number one.", "code": "def one():\n    return 1", "test_list": ["assert one() == 1"]},
]

_EVAL_ROW: dict = {
    "text": "Return the number two.",
    "code": "def two():\n    return 2",
    "test_list": ["assert two() == 2", "assert two() != 0"],
}

# Expected prompt (messages, flat concat, ground truth, completions).
# MBPP_OLMES is normally 3-shot (here reduced to 1 fake shot for readability)
_EXPECTED = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=(
                "Please provide a self-contained Python script that solves the following problem"
                " in a markdown code block:\n```\nReturn the number one.\nassert one() == 1\n```\n"
            ),
        ),
        Message(role=Role.ASSISTANT, content="def one():\n    return 1\n"),
        Message(
            role=Role.USER,
            content=(
                "Please provide a self-contained Python script that solves the following problem"
                " in a markdown code block:\n```\nReturn the number two.\nassert two() == 2\n```\n"
            ),
        ),
        Message(role=Role.ASSISTANT, content="Here is the completed function:\n\n```python\n"),
    ],
    concat="""\
Please provide a self-contained Python script that solves the following problem in a markdown code block:
```
Return the number one.
assert one() == 1
```
def one():
    return 1


Please provide a self-contained Python script that solves the following problem in a markdown code block:
```
Return the number two.
assert two() == 2
```
Here is the completed function:

```python""",
    ground_truth="['assert two() == 2', 'assert two() != 0']",
    completions=None,
)


def _assert_sample_matches(sample, expected: ExpectedPrompt) -> None:
    assert sample.messages == expected.messages
    assert ConcatFormatter().format(sample.messages, output_mode="string") == expected.concat
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.completions


def test_mbpp_olmes_offline_prompt_formatting() -> None:
    def mock_fewshot_examples(self, item):
        return list(_FEWSHOT_EXAMPLES)

    task = MBPP_OLMES.with_overwrite(num_fewshot=3, custom_subjects=[_SUBJECT], custom_hf_revision=None)
    mock_dataset = DatasetDict({task.SAMPLE_SPLIT: Dataset.from_list([_EVAL_ROW])})

    with patch.object(MBPP_OLMES, "_sample_fewshot_examples", mock_fewshot_examples):
        with patch.object(task, "_load_hf_dataset", return_value=mock_dataset):
            sample = next(iter(task.iterate_samples(1)))

    _assert_sample_matches(sample, _EXPECTED)


# ---------------------------------------------------------------------------
# MBPP_EvalPlus
# ---------------------------------------------------------------------------

_EVALPLUS_EXPECTED = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=(
                "Please provide a self-contained Python script that solves the following problem"
                " in a markdown code block:\n```\nReturn the number one.\nassert one() == 1\n```\n"
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "Below is a Python script with a self-contained function that solves the problem"
                " and passes corresponding tests:\n```python\ndef one():\n    return 1\n```"
            ),
        ),
        Message(
            role=Role.USER,
            content=(
                "Please provide a self-contained Python script that solves the following problem"
                " in a markdown code block:\n```\nReturn the number two.\nassert two() == 2\n```\n"
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "Below is a Python script with a self-contained function that solves the problem"
                " and passes corresponding tests:\n```python"
            ),
        ),
    ],
    concat="""\
Please provide a self-contained Python script that solves the following problem in a markdown code block:
```
Return the number one.
assert one() == 1
```
Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
```python
def one():
    return 1
```

Please provide a self-contained Python script that solves the following problem in a markdown code block:
```
Return the number two.
assert two() == 2
```
Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
```python""",
    ground_truth="['assert two() == 2', 'assert two() != 0']",
    completions=None,
)


def test_mbpp_evalplus_offline_prompt_formatting() -> None:
    def mock_fewshot_examples(self, item):
        return list(_FEWSHOT_EXAMPLES)

    task = MBPP_EvalPlus.with_overwrite(num_fewshot=3, custom_subjects=[_SUBJECT], custom_hf_revision=None)
    mock_dataset = DatasetDict({task.SAMPLE_SPLIT: Dataset.from_list([_EVAL_ROW])})

    with patch.object(MBPP_EvalPlus, "_sample_fewshot_examples", mock_fewshot_examples):
        with patch.object(task, "_load_hf_dataset", return_value=mock_dataset):
            sample = next(iter(task.iterate_samples(1)))

    _assert_sample_matches(sample, _EVALPLUS_EXPECTED)


# ---------------------------------------------------------------------------
# MBPP_BPB_EvalPlus
# ---------------------------------------------------------------------------

_BPB_EVALPLUS_EXPECTED = ExpectedPrompt(
    messages=_EVALPLUS_EXPECTED.messages,
    concat=_EVALPLUS_EXPECTED.concat,
    ground_truth="\ndef two():\n    return 2\n```",
    completions=["\ndef two():\n    return 2\n```"],
)


def test_mbpp_bpb_evalplus_offline_prompt_formatting() -> None:
    def mock_fewshot_examples(self, item):
        return list(_FEWSHOT_EXAMPLES)

    task = MBPP_BPB_EvalPlus.with_overwrite(num_fewshot=3, custom_subjects=[_SUBJECT], custom_hf_revision=None)
    mock_dataset = DatasetDict({task.SAMPLE_SPLIT: Dataset.from_list([_EVAL_ROW])})

    with patch.object(MBPP_BPB_EvalPlus, "_sample_fewshot_examples", mock_fewshot_examples):
        with patch.object(task, "_load_hf_dataset", return_value=mock_dataset):
            sample = next(iter(task.iterate_samples(1)))

    _assert_sample_matches(sample, _BPB_EVALPLUS_EXPECTED)
