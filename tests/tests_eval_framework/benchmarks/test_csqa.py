"""Specification of the CommonsenseQA task.

The spec tests build the real benchmarks over a fictional dataset and assert the assembled messages,
ground truth, and scored completions — so this file reads as CommonsenseQA's prompt spec, with
``composed.py`` an implementation detail. ``test_formatter_hash`` separately pins every registered variant
against the real HuggingFace data.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.csqa import CSQA_BENCHMARKS, csqa_mc_olmes
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
