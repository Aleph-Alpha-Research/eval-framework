"""Specification of the Social IQa task.

The spec tests build the real benchmarks over a fictional dataset and assert the assembled messages,
ground truth, and scored completions — so this file reads as Social IQa's prompt spec, with ``composed.py``
an implementation detail. ``test_formatter_hash`` separately pins every registered variant against the
real HuggingFace data.
"""

from collections.abc import Callable
from typing import Any

import pytest

from eval_framework.benchmarks.social_iqa import (
    SOCIAL_IQA_BENCHMARKS,
    social_iqa_bpb,
    social_iqa_bpb_olmes,
    social_iqa_cloze,
    social_iqa_cloze_olmes,
    social_iqa_mc,
    social_iqa_mc_olmes,
)
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


# ---------------------------------------------------------------------------
# Cloze and BPB, OLMES (OLMo 3 Base Easy): validation split, the first k OLMES demonstrations, options hidden
# ---------------------------------------------------------------------------

# The first five OLMES demonstrations, each solved with its full answer text (not a letter).
_RC_OLMES_FIVE_SHOT_MESSAGES = [
    Message(
        role=Role.USER,
        content="Question: Cameron decided to have a barbecue and gathered her friends together."
        " How would Others feel as a result?\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: like attending"),
    Message(
        role=Role.USER,
        content="Question: Quinn wanted to help me clean my room up because it was so messy."
        " What will Quinn want to do next?\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: Pick up the dirty clothes"),
    Message(
        role=Role.USER,
        content="Question: Jan needed to give out jobs for an upcoming project at work."
        " What will Others want to do next?\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: get to work"),
    Message(
        role=Role.USER,
        content="Question: Their cat kept trying to escape out of the window, so Jan placed an obstacle in the way."
        " How would Jan feel afterwards?\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: relieved for fixing the problem"),
    Message(
        role=Role.USER,
        content="Question: Remy was an expert fisherman and was on the water with Kai. Remy baited Kai's hook."
        " What will Remy want to do next?\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: cast the line"),
]
# The eval row's own prompt: context + question only, no option menu.
_RC_EVAL_MESSAGES = [
    Message(role=Role.USER, content="Question: Alex spilled coffee on a stranger. What will Alex want to do next?\n"),
    Message(role=Role.ASSISTANT, content="Answer:"),
]


def test_social_iqa_cloze_olmes_fiveshot_prompt() -> None:
    # Given the real Social IQa cloze (OLMES) benchmark over one fictional row on its sample split (validation)
    benchmark = social_iqa_cloze_olmes(dataset=DatasetStub({"validation": [_EVAL_ROW]}))
    # When we assemble its first sample (five-shot)
    sample = first_sample(benchmark, num_fewshot=5)
    # Then the first five OLMES demonstrations precede the question, and every full answer text is scored:
    assert sample.messages == _RC_OLMES_FIVE_SHOT_MESSAGES + _RC_EVAL_MESSAGES
    assert sample.ground_truth == " Apologise"
    assert sample.possible_completions == [" Apologise", " Laugh", " Walk away"]


def test_social_iqa_bpb_olmes_fiveshot_prompt() -> None:
    # Given the real Social IQa BPB (OLMES) benchmark over one fictional row on its sample split (validation)
    benchmark = social_iqa_bpb_olmes(dataset=DatasetStub({"validation": [_EVAL_ROW]}))
    # When we assemble its first sample (five-shot)
    sample = first_sample(benchmark, num_fewshot=5)
    # Then the prompt is the cloze one, but only the gold answer's full text is scored:
    assert sample.messages == _RC_OLMES_FIVE_SHOT_MESSAGES + _RC_EVAL_MESSAGES
    assert sample.ground_truth == " Apologise"
    assert sample.possible_completions == [" Apologise"]


@pytest.mark.parametrize("make_benchmark", [social_iqa_cloze_olmes, social_iqa_bpb_olmes], ids=["cloze", "bpb"])
def test_social_iqa_rc_olmes_evaluates_the_validation_split(make_benchmark: Callable[..., Benchmark]) -> None:
    # Given the benchmark over a dataset whose train and validation splits hold different rows
    benchmark = make_benchmark(dataset=DatasetStub({"train": [_OTHER_SPLIT_ROW], "validation": [_EVAL_ROW]}))
    # When we assemble its first sample
    sample = first_sample(benchmark, num_fewshot=0)
    # Then it asks the validation row's question:
    assert sample.messages == _RC_EVAL_MESSAGES


# ---------------------------------------------------------------------------
# Generic MC, cloze and BPB: validation split, demonstrations drawn at random from train
# ---------------------------------------------------------------------------

# A dataset whose train split holds the only possible demonstration and whose validation split the eval row.
_TRAIN_DEMO_VALIDATION_EVAL = {"train": [_OTHER_SPLIT_ROW], "validation": [_EVAL_ROW]}


def test_social_iqa_mc_oneshot_prompt() -> None:
    # Given the real generic Social IQa MC benchmark over a train demonstration and a validation row
    benchmark = social_iqa_mc(dataset=DatasetStub(_TRAIN_DEMO_VALIDATION_EVAL))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the train row, solved with its letter, precedes the validation row; options are lettered choices:
    assert sample.messages == [
        Message(
            role=Role.USER,
            content="Question: Jordan studied all night for the exam. How would Jordan feel afterwards?\n"
            "A. Careless\nB. Tired\nC. Bored\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer: B"),
        Message(
            role=Role.USER,
            content="Question: Alex spilled coffee on a stranger. What will Alex want to do next?\n"
            "A. Apologise\nB. Laugh\nC. Walk away\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " A"
    assert sample.possible_completions == [" A", " B", " C"]


# The train row solved with its full answer text, then the validation row's question without options.
_RC_ONE_SHOT_MESSAGES = [
    Message(
        role=Role.USER, content="Question: Jordan studied all night for the exam. How would Jordan feel afterwards?\n"
    ),
    Message(role=Role.ASSISTANT, content="Answer: Tired"),
    Message(role=Role.USER, content="Question: Alex spilled coffee on a stranger. What will Alex want to do next?\n"),
    Message(role=Role.ASSISTANT, content="Answer:"),
]


def test_social_iqa_cloze_oneshot_prompt() -> None:
    # Given the real generic Social IQa cloze benchmark over a train demonstration and a validation row
    benchmark = social_iqa_cloze(dataset=DatasetStub(_TRAIN_DEMO_VALIDATION_EVAL))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the train row precedes the validation row's question, and every full answer text is scored:
    assert sample.messages == _RC_ONE_SHOT_MESSAGES
    assert sample.ground_truth == " Apologise"
    assert sample.possible_completions == [" Apologise", " Laugh", " Walk away"]


def test_social_iqa_bpb_oneshot_prompt() -> None:
    # Given the real generic Social IQa BPB benchmark over a train demonstration and a validation row
    benchmark = social_iqa_bpb(dataset=DatasetStub(_TRAIN_DEMO_VALIDATION_EVAL))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the prompt is the cloze one, but only the gold answer's full text is scored:
    assert sample.messages == _RC_ONE_SHOT_MESSAGES
    assert sample.ground_truth == " Apologise"
    assert sample.possible_completions == [" Apologise"]
