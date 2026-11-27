"""Specification of the CommonsenseQA task.

The spec tests build the real benchmarks over a fictional dataset and assert the assembled messages,
ground truth, and scored completions — so this file reads as CommonsenseQA's prompt spec, with
``composed.py`` an implementation detail. ``test_formatter_hash`` separately pins every registered variant
against the real HuggingFace data.
"""

from collections.abc import Callable
from typing import Any

import pytest

from eval_framework.benchmarks.csqa import (
    CSQA_BENCHMARKS,
    csqa_bpb,
    csqa_bpb_olmes,
    csqa_cloze,
    csqa_cloze_olmes,
    csqa_mc,
    csqa_mc_olmes,
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
@pytest.mark.parametrize("benchmark", CSQA_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the CommonsenseQA format (NOT real dataset examples): a question, five labelled options
# (the nested ``choices`` mapping), and the ``answerKey`` label. Options are shown in fixed order (no shuffle).
_EVAL_ROW: dict[str, Any] = {
    "question": "Where would you store a pillow you are not using?",
    "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["bedroom", "cupboard", "floor", "kitchen", "car"]},
    "answerKey": "B",
}
# A row placed in the split a variant must NOT evaluate.
_OTHER_SPLIT_ROW: dict[str, Any] = {
    "question": "What do people do when they are tired?",
    "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["run", "sleep", "eat", "sing", "swim"]},
    "answerKey": "B",
}

# ---------------------------------------------------------------------------
# MC, OLMES (OLMo 3 Base Main): train split, the first k OLMES demonstrations, lettered options, letters scored
# ---------------------------------------------------------------------------

# The first five OLMES demonstrations, each with its option menu and solved with the correct letter.
_MC_OLMES_FIVE_SHOT_MESSAGES = [
    Message(
        role=Role.USER,
        content="Question: Sammy wanted to go to where the people were.  Where might he go?\n"
        " A. race track\n B. populated areas\n C. the desert\n D. apartment\n E. roadblock\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: B"),
    Message(
        role=Role.USER,
        content="Question: Google Maps and other highway and street GPS services have replaced what?\n"
        " A. united states\n B. mexico\n C. countryside\n D. atlas\n E. oceans\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: D"),
    Message(
        role=Role.USER,
        content="Question: The fox walked from the city into the forest, what was it looking for?\n"
        " A. pretty flowers.\n B. hen house\n C. natural habitat\n D. storybook\n E. dense forest\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: C"),
    Message(
        role=Role.USER,
        content="Question: What is it called when you slowly cook using a grill?\n"
        " A. backyard\n B. restaurant\n C. crockpot\n D. neighbor's house\n E. barbeque\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: E"),
    Message(
        role=Role.USER,
        content="Question: What would you do if you want to be able to earn money?\n"
        " A. apply for job\n B. stand in line\n C. take care of proposals\n D. pass course\n E. play the lottery\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: A"),
]
# The eval row's own prompt: the question with its option menu.
_MC_EVAL_MESSAGES = [
    Message(
        role=Role.USER,
        content="Question: Where would you store a pillow you are not using?\n"
        " A. bedroom\n B. cupboard\n C. floor\n D. kitchen\n E. car\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer:"),
]


def test_csqa_mc_olmes_zeroshot_prompt() -> None:
    # Given the real CommonsenseQA MC (OLMES) benchmark over one fictional row on its sample split (train)
    benchmark = csqa_mc_olmes(dataset=DatasetStub({"train": [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then options are space-prefixed lettered choices, and the letters are scored:
    assert sample.messages == _MC_EVAL_MESSAGES
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [" A", " B", " C", " D", " E"]


def test_csqa_mc_olmes_fiveshot_prompt() -> None:
    # Given the real CommonsenseQA MC (OLMES) benchmark over one fictional row on its sample split (train)
    benchmark = csqa_mc_olmes(dataset=DatasetStub({"train": [_EVAL_ROW]}))
    # When we assemble its first sample (five-shot)
    sample = first_sample(benchmark, num_fewshot=5)
    # Then the first five OLMES demonstrations, each solved with its letter, precede the eval row's prompt:
    assert sample.messages == _MC_OLMES_FIVE_SHOT_MESSAGES + _MC_EVAL_MESSAGES
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [" A", " B", " C", " D", " E"]


def test_csqa_mc_olmes_evaluates_the_train_split() -> None:
    # Given the benchmark over a dataset whose train and validation splits hold different rows
    benchmark = csqa_mc_olmes(dataset=DatasetStub({"train": [_EVAL_ROW], "validation": [_OTHER_SPLIT_ROW]}))
    # When we assemble its first sample
    sample = first_sample(benchmark, num_fewshot=0)
    # Then it asks the train row's question:
    assert sample.messages == _MC_EVAL_MESSAGES


# ---------------------------------------------------------------------------
# Cloze and BPB, OLMES (OLMo 3 Base Easy): validation split, the first k OLMES demonstrations, options hidden
# ---------------------------------------------------------------------------

# The first five OLMES demonstrations, each solved with its full answer text (not a letter).
_RC_OLMES_FIVE_SHOT_MESSAGES = [
    Message(role=Role.USER, content="Question: Sammy wanted to go to where the people were.  Where might he go?\n"),
    Message(role=Role.ASSISTANT, content="Answer: populated areas"),
    Message(
        role=Role.USER, content="Question: Google Maps and other highway and street GPS services have replaced what?\n"
    ),
    Message(role=Role.ASSISTANT, content="Answer: atlas"),
    Message(
        role=Role.USER, content="Question: The fox walked from the city into the forest, what was it looking for?\n"
    ),
    Message(role=Role.ASSISTANT, content="Answer: natural habitat"),
    Message(role=Role.USER, content="Question: What is it called when you slowly cook using a grill?\n"),
    Message(role=Role.ASSISTANT, content="Answer: barbeque"),
    Message(role=Role.USER, content="Question: What would you do if you want to be able to earn money?\n"),
    Message(role=Role.ASSISTANT, content="Answer: apply for job"),
]
# The eval row's own prompt: the question only, no option menu.
_RC_EVAL_MESSAGES = [
    Message(role=Role.USER, content="Question: Where would you store a pillow you are not using?\n"),
    Message(role=Role.ASSISTANT, content="Answer:"),
]


def test_csqa_cloze_olmes_fiveshot_prompt() -> None:
    # Given the real CommonsenseQA cloze (OLMES) benchmark over one fictional row on its sample split (validation)
    benchmark = csqa_cloze_olmes(dataset=DatasetStub({"validation": [_EVAL_ROW]}))
    # When we assemble its first sample (five-shot)
    sample = first_sample(benchmark, num_fewshot=5)
    # Then the first five OLMES demonstrations precede the question, and every full answer text is scored:
    assert sample.messages == _RC_OLMES_FIVE_SHOT_MESSAGES + _RC_EVAL_MESSAGES
    assert sample.ground_truth == " cupboard"
    assert sample.possible_completions == [" bedroom", " cupboard", " floor", " kitchen", " car"]


def test_csqa_bpb_olmes_fiveshot_prompt() -> None:
    # Given the real CommonsenseQA BPB (OLMES) benchmark over one fictional row on its sample split (validation)
    benchmark = csqa_bpb_olmes(dataset=DatasetStub({"validation": [_EVAL_ROW]}))
    # When we assemble its first sample (five-shot)
    sample = first_sample(benchmark, num_fewshot=5)
    # Then the prompt is the cloze one, but only the gold answer's full text is scored:
    assert sample.messages == _RC_OLMES_FIVE_SHOT_MESSAGES + _RC_EVAL_MESSAGES
    assert sample.ground_truth == " cupboard"
    assert sample.possible_completions == [" cupboard"]


@pytest.mark.parametrize("make_benchmark", [csqa_cloze_olmes, csqa_bpb_olmes], ids=["cloze", "bpb"])
def test_csqa_rc_olmes_evaluates_the_validation_split(make_benchmark: Callable[..., Benchmark]) -> None:
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


def test_csqa_mc_oneshot_prompt() -> None:
    # Given the real generic CommonsenseQA MC benchmark over a train demonstration and a validation row
    benchmark = csqa_mc(dataset=DatasetStub(_TRAIN_DEMO_VALIDATION_EVAL))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the train row, solved with its letter, precedes the validation row; options are lettered choices:
    assert sample.messages == [
        Message(
            role=Role.USER,
            content="Question: What do people do when they are tired?\nA. run\nB. sleep\nC. eat\nD. sing\nE. swim\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer: B"),
        Message(
            role=Role.USER,
            content="Question: Where would you store a pillow you are not using?\n"
            "A. bedroom\nB. cupboard\nC. floor\nD. kitchen\nE. car\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [" A", " B", " C", " D", " E"]


# The train row solved with its full answer text, then the validation row's question without options.
_RC_ONE_SHOT_MESSAGES = [
    Message(role=Role.USER, content="Question: What do people do when they are tired?\n"),
    Message(role=Role.ASSISTANT, content="Answer: sleep"),
    Message(role=Role.USER, content="Question: Where would you store a pillow you are not using?\n"),
    Message(role=Role.ASSISTANT, content="Answer:"),
]


def test_csqa_cloze_oneshot_prompt() -> None:
    # Given the real generic CommonsenseQA cloze benchmark over a train demonstration and a validation row
    benchmark = csqa_cloze(dataset=DatasetStub(_TRAIN_DEMO_VALIDATION_EVAL))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the train row precedes the validation row's question, and every full answer text is scored:
    assert sample.messages == _RC_ONE_SHOT_MESSAGES
    assert sample.ground_truth == " cupboard"
    assert sample.possible_completions == [" bedroom", " cupboard", " floor", " kitchen", " car"]


def test_csqa_bpb_oneshot_prompt() -> None:
    # Given the real generic CommonsenseQA BPB benchmark over a train demonstration and a validation row
    benchmark = csqa_bpb(dataset=DatasetStub(_TRAIN_DEMO_VALIDATION_EVAL))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the prompt is the cloze one, but only the gold answer's full text is scored:
    assert sample.messages == _RC_ONE_SHOT_MESSAGES
    assert sample.ground_truth == " cupboard"
    assert sample.possible_completions == [" cupboard"]
