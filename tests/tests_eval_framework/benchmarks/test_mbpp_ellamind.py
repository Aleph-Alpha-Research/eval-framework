"""Specification of the composed German MBPP (EllaMind) tasks.

These reuse the ``mbpp`` builders/reconstructs with a German instruction wrapper and same-split few-shot
sampling. ``test_formatter_hash`` pins each against the real data (1-shot, as recorded); the offline tests pin
the assembled German prompt / candidates against a stub dataset.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.mbpp_ellamind import (
    MBPP_ELLAMIND_BENCHMARKS,
    mbpp_de_bpb_olmes,
    mbpp_de_evalplus_instruct,
    mbpp_de_olmes,
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
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import assert_benchmark_formatter_hash


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("benchmark", MBPP_ELLAMIND_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)  # recorded at the 1-shot default


_ROW: dict[str, Any] = {
    "text": "Schreibe eine Funktion, die zwei Zahlen addiert.",
    "code": "def add(a, b):\n    return a + b",
    "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
}


def _stub() -> DatasetStub:
    return DatasetStub({"test": [_ROW]})


def test_de_olmes_prompt() -> None:
    sample = first_sample(mbpp_de_olmes(dataset=_stub()), num_fewshot=0)
    assert sample.messages[0] == Message(
        role=Role.USER,
        content=(
            "Bitte erstelle ein in sich geschlossenes Python-Skript, das das folgende Problem in einem "
            "Markdown-Code-Block löst:\n```\nSchreibe eine Funktion, die zwei Zahlen addiert.\n"
            "assert add(1, 2) == 3\n```\n"
        ),
    )
    assert sample.messages[1] == Message(role=Role.ASSISTANT, content="Hier ist die fertige Funktion:\n\n```python\n")
    assert sample.ground_truth == "['assert add(1, 2) == 3', 'assert add(0, 0) == 0']"


def test_de_bpb_olmes_scores_fenced_solution() -> None:
    sample = first_sample(mbpp_de_bpb_olmes(dataset=_stub()), num_fewshot=0)
    gold = _ROW["code"] + "\n```"
    assert sample.messages[-1] == Message(
        role=Role.USER, content=_ROW["text"] + "\n```python\n"
    )  # no assistant cue for this BPB variant
    assert sample.possible_completions == [gold]
    assert sample.ground_truth == gold


def test_de_instruct_prompt_has_no_cue() -> None:
    sample = first_sample(mbpp_de_evalplus_instruct(dataset=_stub()), num_fewshot=0)
    assert all(m.role != Role.ASSISTANT for m in sample.messages)  # cue="" → no assistant turn
    assert sample.messages[-1].role == Role.USER
    assert sample.messages[-1].content.startswith("Bitte erstelle ein in sich geschlossenes Python-Skript,")
