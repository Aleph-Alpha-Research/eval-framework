"""Specification of the composed German HumanEval (EllaMind) tasks.

These reuse the ``humaneval`` builders retargeted to the German ``deu`` subject. ``test_formatter_hash`` pins
each variant against the real HuggingFace data (1-shot, as recorded); the offline tests pin the assembled
prompt / candidates against a stub, and ``test_instruct_extracts_function`` pins the markdown-block extraction
that ``HumanEvalDEInstruct`` relies on.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.humaneval import CODE_TO_EXECUTE, HumanEvalMetricContext
from eval_framework.benchmarks.humaneval_ellamind import (
    HUMANEVAL_ELLAMIND_BENCHMARKS,
    _instruct_reconstruct,
    humaneval_de_bpb_olmes,
    humaneval_de_bpb_olmes_v2,
    humaneval_de_instruct,
    humaneval_de_olmes,
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
@pytest.mark.parametrize("benchmark", HUMANEVAL_ELLAMIND_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)  # recorded at the 1-shot default


# ---------------------------------------------------------------------------
# Offline prompt / candidate specs
# ---------------------------------------------------------------------------

_ROW: dict[str, Any] = {
    "prompt": '\n\ndef add(a: int, b: int) -> int:\n    """Addiert zwei Zahlen."""\n',
    "canonical_solution": "    return a + b",
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    "entry_point": "add",
}


def _stub() -> DatasetStub:
    return DatasetStub({"test": [_ROW]})


def test_de_olmes_prompt_and_context() -> None:
    sample = first_sample(humaneval_de_olmes(dataset=_stub()), num_fewshot=0)
    assert sample.messages == [Message(role=Role.USER, content="```python\n" + _ROW["prompt"].rstrip() + "\n")]
    assert sample.ground_truth == "Success"
    assert sample.possible_completions is None
    assert sample.context == HumanEvalMetricContext(test=_ROW["test"], entry_point="add", prompt=_ROW["prompt"])


def test_de_bpb_olmes_scores_bare_solution() -> None:
    # The raw prompt (styler adds the trailing newline), scoring the un-fenced gold solution.
    sample = first_sample(humaneval_de_bpb_olmes(dataset=_stub()), num_fewshot=0)
    assert sample.messages == [Message(role=Role.USER, content=_ROW["prompt"].strip("\n") + "\n")]
    assert sample.possible_completions == [_ROW["canonical_solution"]]
    assert sample.ground_truth == _ROW["canonical_solution"]


def test_de_bpb_olmes_v2_scores_fenced_solution() -> None:
    sample = first_sample(humaneval_de_bpb_olmes_v2(dataset=_stub()), num_fewshot=0)
    gold = _ROW["canonical_solution"].rstrip() + "\n```"
    assert sample.messages == [Message(role=Role.USER, content="```python\n" + _ROW["prompt"].rstrip() + "\n")]
    assert sample.possible_completions == [gold]
    assert sample.ground_truth == gold


def test_de_instruct_prompt() -> None:
    sample = first_sample(humaneval_de_instruct(dataset=_stub()), num_fewshot=0)
    assert sample.messages == [
        Message(
            role=Role.USER,
            content=(
                "Vervollständige die folgende Python-Funktion. Gib ausschließlich die vollständige Funktion "
                f"in einem Markdown-Codeblock zurück:\n```python\n{_ROW['prompt'].strip()}\n```\n"
            ),
        )
    ]
    assert sample.ground_truth == "Success"


def test_instruct_extracts_function() -> None:
    # The instruct model answers in free-form German prose; reconstruction pulls the function out of the
    # markdown block and drops the surrounding text before splicing it into the runnable program.
    context = HumanEvalMetricContext(test=_ROW["test"], entry_point="add", prompt=_ROW["prompt"])
    code = _instruct_reconstruct(
        "Hier ist die Lösung:\n```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```",
        context=context,
        ground_truth="Success",
        messages=[],
    )
    assert "Hier ist die Lösung" not in code
    assert code.count("def add(a: int, b: int) -> int:") == 1
    assert "check(add)" in code
    assert code == CODE_TO_EXECUTE.format(
        start_of_code="",
        completion_text="def add(a: int, b: int) -> int:\n    return a + b",
        test_code=_ROW["test"],
        entry_point="add",
    )
