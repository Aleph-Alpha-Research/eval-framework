"""Specification of the composed LBPP tasks.

``test_formatter_hash`` pins each variant against the real data; the offline tests pin the assembled prompt,
ground truth and BPB candidate, the reconstructed test program, and the safe decoding against fictional rows.
"""

import base64
import json
import pickle
import subprocess
import sys
import zlib
from typing import Any

import pytest

from eval_framework.benchmarks.lbpp import LBPP_BENCHMARKS, lbpp, lbpp_bpb
from eval_framework.contract import Benchmark
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    NoStripConcatFormatter,
    Role,
)
from tests.tests_eval_framework.benchmarks.utils import (
    DatasetStub,
    StubLlm,
    assert_benchmark_formatter_hash,
    first_sample,
)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("benchmark", LBPP_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls, num_fewshot=1)


def _encoded(value: Any) -> str:
    """LBPP's encoding of its code columns: ``base64(zlib(pickle(json)))``."""
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(value)))).decode()


# Fictional rows following the CohereLabs/lbpp schema. NOT real problems from the dataset: the setup imports a
# module before its ``from code import`` line, and the second test spans a multi-line string literal.
_EVAL_ROW = {
    "task_id": "lbpp/python/900",
    "instruction": "Write a function `shout(text: str) -> str` that upper-cases a text.\n",
    "completion": _encoded("def shout(text: str) -> str:\n    return text.upper()\n"),
    "test_setup": _encoded("import string\nfrom code import shout\n\n"),
    "test_list": _encoded(
        [
            "assert shout(string.ascii_lowercase) == string.ascii_uppercase",
            'poem = """roses\nviolets"""\nassert shout(poem) == "ROSES\\nVIOLETS"',
        ]
    ),
    "test_file": _encoded(""),
}
_CANARY_ROW = {**_EVAL_ROW, "task_id": "lbpp/python/042", "instruction": "Return the canary string."}


def test_the_prompt_shows_the_instruction_and_first_test_then_cues_a_python_script() -> None:
    # Given one problem
    # When the 0-shot sample is assembled
    sample = first_sample(lbpp(dataset=DatasetStub({"test": [_EVAL_ROW]})), num_fewshot=0)

    # Then the stripped instruction and the first test sit in a markdown block, answered after the EvalPlus cue
    assert sample.messages == [
        Message(
            role=Role.USER,
            content=(
                "Please provide a self-contained Python script that solves the following problem in a markdown"
                " code block:\n```\nWrite a function `shout(text: str) -> str` that upper-cases a text.\n"
                "assert shout(string.ascii_lowercase) == string.ascii_uppercase\n```\n"
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "Below is a Python script with a self-contained function that solves the problem and passes"
                " corresponding tests:\n```python"
            ),
        ),
    ]


def test_the_ground_truth_is_the_setup_without_its_code_import_then_the_tests() -> None:
    sample = first_sample(lbpp(dataset=DatasetStub({"test": [_EVAL_ROW]})), num_fewshot=0)
    assert sample.ground_truth == (
        "import string\n\n"
        "assert shout(string.ascii_lowercase) == string.ascii_uppercase\n"
        'poem = """roses\nviolets"""\nassert shout(poem) == "ROSES\\nVIOLETS"'
    )


def test_the_canary_row_is_not_evaluated() -> None:
    # Given the leakage canary next to one problem
    benchmark = lbpp(dataset=DatasetStub({"test": [_CANARY_ROW, _EVAL_ROW]}))

    # When all samples are assembled
    samples = list(benchmark.create(num_fewshot=0, custom_subjects=None, custom_hf_revision=None).iterate_samples())

    # Then only the problem is evaluated
    assert [sample.messages[0].content for sample in samples] == [
        first_sample(lbpp(dataset=DatasetStub({"test": [_EVAL_ROW]})), num_fewshot=0).messages[0].content
    ]


def _verdict(model_output: str) -> str:
    """What the program reconstructed from ``model_output`` prints last when run (the metric scores ``True``)."""
    evaluation = lbpp(dataset=DatasetStub({"test": [_EVAL_ROW]})).create(
        num_fewshot=0, custom_subjects=None, custom_hf_revision=None
    )
    sample = next(iter(evaluation.iterate_samples(1)))
    (completion,) = evaluation.generate_completions(
        StubLlm(model_output), [sample], stop_sequences=evaluation.get_stop_sequences()
    )
    run = subprocess.run([sys.executable, "-c", completion.completion], capture_output=True, text=True, timeout=30)
    return run.stdout.split()[-1]


def test_a_correct_script_passes_the_tests() -> None:
    # Given a correct solution, closing its fence and rambling on (the runner forwards the stop sequences)
    # When its reconstructed program runs
    # Then the tests pass, the multi-line literal intact and the setup's import kept
    assert _verdict("\ndef shout(text: str) -> str:\n    return text.upper()\n```\n\nThat's it!") == "True"


def test_a_wrong_script_fails_even_when_it_prints_true_itself() -> None:
    assert _verdict("\ndef shout(text: str) -> str:\n    return text\nprint(True)\n```") == "False"


def test_the_bpb_candidate_is_the_gold_solution_closing_its_fence() -> None:
    # Given one problem
    # When the 0-shot BPB sample is assembled
    sample = first_sample(lbpp_bpb(dataset=DatasetStub({"test": [_EVAL_ROW]})), num_fewshot=0)

    # Then the stripped gold solution, closing the cue's fence, is the sole candidate
    assert sample.possible_completions == ["\ndef shout(text: str) -> str:\n    return text.upper()\n```"]


def test_a_field_pickling_an_object_is_refused() -> None:
    # Given a tampered row whose tests unpickle to a global (here a harmless builtin) instead of a plain value
    tampered = {**_EVAL_ROW, "test_list": base64.b64encode(zlib.compress(pickle.dumps(print))).decode()}

    # When the sample is assembled, then decoding refuses it
    with pytest.raises(pickle.UnpicklingError, match="builtins.print"):
        first_sample(lbpp(dataset=DatasetStub({"test": [tampered]})), num_fewshot=0)
