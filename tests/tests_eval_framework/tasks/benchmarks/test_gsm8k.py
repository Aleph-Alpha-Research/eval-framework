from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from eval_framework.tasks.benchmarks.gsm8k import GSM8KBPB
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.tasks.benchmarks.utils import (
    ExpectedPrompt,
    get_task_names_for_module,
    run_formatter_hash_test,
)

_NUM_FEWSHOT = {"GSM8K_OLMES": 8}


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", get_task_names_for_module("gsm8k"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, num_fewshot=_NUM_FEWSHOT.get(task_name, 1))


_SUBJECT = "main"

_EVAL_ROW: dict[str, str] = {
    "question": "If there are 2 apples and 3 oranges, how many fruits are there?",
    "answer": "There are 2 apples and 3 oranges. So there are 2 + 3 = 5 fruits. #### 5",
}

_FEWSHOT_ROW: dict[str, str] = {
    "question": "If there are 5 cats and 2 dogs, how many pets are there?",
    "answer": "There are 5 cats and 2 dogs. So there are 5 + 2 = 7 pets. #### 7",
}

_CUE = "Answer:"
# _get_instruction_text no longer embeds the cue; it becomes its own ASSISTANT message.
_EVAL_Q = f"Question: {_EVAL_ROW['question']}\n"
_FEWSHOT_Q = f"Question: {_FEWSHOT_ROW['question']}\n"

_EVAL_COMPLETION = " There are 2 apples and 3 oranges. So there are 2 + 3 = 5 fruits. So the answer is 5."
_FEWSHOT_ANSWER = f"{_CUE} There are 5 cats and 2 dogs. So there are 5 + 2 = 7 pets. So the answer is 7."
_GROUND_TRUTH = _EVAL_COMPLETION
_COMPLETIONS = [_EVAL_COMPLETION]

# Concat strings are identical to before; structure just moves the cue to an ASSISTANT turn.
_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=_EVAL_Q),
        Message(role=Role.ASSISTANT, content=_CUE),
    ],
    concat=f"{_EVAL_Q}{_CUE}",
    ground_truth=_GROUND_TRUTH,
    completions=_COMPLETIONS,
)

_ONESHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=_FEWSHOT_Q),
        Message(role=Role.ASSISTANT, content=_FEWSHOT_ANSWER),
        Message(role=Role.USER, content=_EVAL_Q),
        Message(role=Role.ASSISTANT, content=_CUE),
    ],
    concat=f"{_FEWSHOT_Q}{_FEWSHOT_ANSWER}\n\n{_EVAL_Q}{_CUE}",
    ground_truth=_GROUND_TRUTH,
    completions=_COMPLETIONS,
)


def _assert_sample_matches(sample, expected: ExpectedPrompt) -> None:
    assert sample.messages == expected.messages
    assert ConcatFormatter().format(sample.messages, output_mode="string") == expected.concat
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.completions


def test_gsm8kbpb_offline_prompt_formatting_zeroshot() -> None:
    def mock_zero_fewshot_examples(self, item):
        return []

    task = GSM8KBPB.with_overwrite(num_fewshot=0, custom_subjects=[_SUBJECT], custom_hf_revision=None)
    mock_dataset = DatasetDict({task.SAMPLE_SPLIT: Dataset.from_list([_EVAL_ROW])})

    with patch.object(GSM8KBPB, "_sample_fewshot_examples", mock_zero_fewshot_examples):
        with patch.object(task, "_load_hf_dataset", return_value=mock_dataset):
            sample = next(iter(task.iterate_samples(1)))

    _assert_sample_matches(sample, _ZEROSHOT)


def test_gsm8kbpb_offline_prompt_formatting_oneshot() -> None:
    def mock_one_fewshot_example(self, item):
        return [_FEWSHOT_ROW]

    task = GSM8KBPB.with_overwrite(num_fewshot=1, custom_subjects=[_SUBJECT], custom_hf_revision=None)
    mock_dataset = DatasetDict({task.SAMPLE_SPLIT: Dataset.from_list([_EVAL_ROW])})

    with patch.object(GSM8KBPB, "_sample_fewshot_examples", mock_one_fewshot_example):
        with patch.object(task, "_load_hf_dataset", return_value=mock_dataset):
            sample = next(iter(task.iterate_samples(1)))

    _assert_sample_matches(sample, _ONESHOT)


def test_gsm8kbpb_loglikelihood_keys_match_ground_truth() -> None:
    def mock_zero_fewshot_examples(self, item):
        return []

    with patch.object(GSM8KBPB, "_sample_fewshot_examples", mock_zero_fewshot_examples):
        task = GSM8KBPB.with_overwrite(num_fewshot=0, custom_subjects=[_SUBJECT], custom_hf_revision=None)
        mock_dataset = DatasetDict({task.SAMPLE_SPLIT: Dataset.from_list([_EVAL_ROW])})
        with patch.object(task, "_load_hf_dataset", return_value=mock_dataset):
            sample = next(iter(task.iterate_samples(1)))
            assert sample.ground_truth in sample.possible_completions
