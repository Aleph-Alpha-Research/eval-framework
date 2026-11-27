"""Specification of the Social IQa task.

The spec tests build the real benchmarks over a fictional dataset and assert the assembled messages,
ground truth, and scored completions — so this file reads as Social IQa's prompt spec, with ``composed.py``
an implementation detail. ``test_formatter_hash`` separately pins every registered variant against the
real HuggingFace data.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.social_iqa import SOCIAL_IQA_BENCHMARKS, social_iqa_mc_olmes
from eval_framework.contract import Benchmark
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    NoStripConcatFormatter,
    Role,
)
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, assert_benchmark_formatter_hash, first_sample


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("benchmark", SOCIAL_IQA_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the Social IQa format (NOT real dataset examples): a context + question, three answer
# options, and a 1-indexed label. Options are shown in fixed order (no shuffle). The shown question joins
# the context and the question with a single space.
_EVAL_ROW: dict[str, Any] = {
    "context": "Alex spilled coffee on a stranger.",
    "question": "What will Alex want to do next?",
    "answerA": "Apologise",
    "answerB": "Laugh",
    "answerC": "Walk away",
    "label": "1",
}
# A row placed in the split a variant must NOT evaluate.
_OTHER_SPLIT_ROW: dict[str, Any] = {
    "context": "Jordan studied all night for the exam.",
    "question": "How would Jordan feel afterwards?",
    "answerA": "Careless",
    "answerB": "Tired",
    "answerC": "Bored",
    "label": "2",
}

# ---------------------------------------------------------------------------
# MC, OLMES (OLMo 3 Base Main): train split, the first k OLMES demonstrations, lettered options, letters scored
# ---------------------------------------------------------------------------

# The first five OLMES demonstrations, each with its option menu and solved with the correct letter.
_MC_OLMES_FIVE_SHOT_MESSAGES = [
    Message(
        role=Role.USER,
        content="Question: Cameron decided to have a barbecue and gathered her friends together."
        " How would Others feel as a result?\n A. like attending\n B. like staying home\n C. a good friend to have\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: A"),
    Message(
        role=Role.USER,
        content="Question: Quinn wanted to help me clean my room up because it was so messy."
        " What will Quinn want to do next?\n"
        " A. Eat messy snacks\n B. help out a friend\n C. Pick up the dirty clothes\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: C"),
    Message(
        role=Role.USER,
        content="Question: Jan needed to give out jobs for an upcoming project at work."
        " What will Others want to do next?\n A. disagree with Jan\n B. get to work\n C. argue with the assignments\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: B"),
    Message(
        role=Role.USER,
        content="Question: Their cat kept trying to escape out of the window, so Jan placed an obstacle in the way."
        " How would Jan feel afterwards?\n"
        " A. scared of losing the cat\n B. normal\n C. relieved for fixing the problem\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: C"),
    Message(
        role=Role.USER,
        content="Question: Remy was an expert fisherman and was on the water with Kai. Remy baited Kai's hook."
        " What will Remy want to do next?\n"
        " A. cast the line\n B. put the boat in the water\n C. invite Kai out on the boat\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: A"),
]
# The eval row's own prompt: context + question with its option menu.
_MC_OLMES_EVAL_MESSAGES = [
    Message(
        role=Role.USER,
        content="Question: Alex spilled coffee on a stranger. What will Alex want to do next?\n"
        " A. Apologise\n B. Laugh\n C. Walk away\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer:"),
]


def test_social_iqa_mc_olmes_zeroshot_prompt() -> None:
    # Given the real Social IQa MC (OLMES) benchmark over one fictional row on its sample split (train)
    benchmark = social_iqa_mc_olmes(dataset=DatasetStub({"train": [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then the context+question is the shown question, options are space-prefixed letters, letters are scored:
    assert sample.messages == _MC_OLMES_EVAL_MESSAGES
    assert sample.ground_truth == " A"
    assert sample.possible_completions == [" A", " B", " C"]


def test_social_iqa_mc_olmes_fiveshot_prompt() -> None:
    # Given the real Social IQa MC (OLMES) benchmark over one fictional row on its sample split (train)
    benchmark = social_iqa_mc_olmes(dataset=DatasetStub({"train": [_EVAL_ROW]}))
    # When we assemble its first sample (five-shot)
    sample = first_sample(benchmark, num_fewshot=5)
    # Then the first five OLMES demonstrations, each solved with its letter, precede the eval row's prompt:
    assert sample.messages == _MC_OLMES_FIVE_SHOT_MESSAGES + _MC_OLMES_EVAL_MESSAGES
    assert sample.ground_truth == " A"
    assert sample.possible_completions == [" A", " B", " C"]


def test_social_iqa_mc_olmes_evaluates_the_train_split() -> None:
    # Given the benchmark over a dataset whose train and validation splits hold different rows
    benchmark = social_iqa_mc_olmes(dataset=DatasetStub({"train": [_EVAL_ROW], "validation": [_OTHER_SPLIT_ROW]}))
    # When we assemble its first sample
    sample = first_sample(benchmark, num_fewshot=0)
    # Then it asks the train row's question:
    assert sample.messages == _MC_OLMES_EVAL_MESSAGES
