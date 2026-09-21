from typing import Any

import pytest

from eval_framework.benchmarks.math_reasoning import (
    _GSM8K_REASONING_QUERY_TEMPLATE,
    _QUERY_TEMPLATE,
    MATH_REASONING_BENCHMARKS,
    _boxed_extractor,
    _gsm8k_reasoning_extractor,
    _strip_string_with_bug,
    gsm8k_reasoning,
    math500_with_bug,
)
from eval_framework.contract import Benchmark
from eval_framework.metrics.completion.minerva_math_utils import strip_string_hendrycks
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

# Each task's canonical shot count, looked up strictly (a new task must declare its count here) so no task
# defaults to 1 and leans on the 0-shot fallback. Every variant here is 0-shot.
_NUM_FEWSHOT = {
    "MATH500": 0,
    "MATH500_V2": 0,
    "AIME2024": 0,
    "AIME2025": 0,
    "AIME2026": 0,
    "GSM8KReasoning": 0,
}


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("benchmark", MATH_REASONING_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls, num_fewshot=_NUM_FEWSHOT[benchmark.id()])


# ---------------------------------------------------------------------------
# Prompt spec (both variants share the prompt)
# ---------------------------------------------------------------------------

_EVAL_ROW: dict[str, Any] = {"problem": "What is 2 + 2?", "answer": "4"}


def test_prompt_is_the_query_template() -> None:
    benchmark = math500_with_bug(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == [Message(role=Role.USER, content=_QUERY_TEMPLATE.format(Question="What is 2 + 2?"))]
    assert sample.ground_truth == "4"
    assert sample.possible_completions is None  # free-form generation


# ---------------------------------------------------------------------------
# Extraction: V1 preserves the normalisation bug, V2 fixes it (both boxed + Answer-line fallback)
# ---------------------------------------------------------------------------

_v1 = _boxed_extractor(_strip_string_with_bug, answer_line_fallback=True)
_v2 = _boxed_extractor(strip_string_hendrycks, answer_line_fallback=True)


@pytest.mark.parametrize(
    "completion, expected",
    [
        (r"So \boxed{42}", "42"),
        (r"\boxed{\frac{1}{2}}", "\\frac{1}{2}"),
        ("Answer: 5", "5"),  # falls back to the Answer: line
        (r"Answer: \boxed{3}", "3"),  # boxed wins over the Answer: line
        (r"\fbox{7}", "[no_answer]"),  # \fbox is not accepted; no Answer: line either
        ("no answer at all", "[no_answer]"),
    ],
)
def test_extraction_shared(completion: str, expected: str) -> None:
    assert _v1(completion) == expected
    assert _v2(completion) == expected


@pytest.mark.parametrize(
    "completion, v1_expected, v2_expected",
    [
        (r"\boxed{\left(3,4\right)}", "\\left(3,4\\right)", "(3,4)"),
        (r"\boxed{90^\circ}", "90^\\circ", "90"),
        (r"\boxed{\frac{1}{2}\!}", "\\frac{1}{2}\\!", "\\frac{1}{2}"),
    ],
)
def test_extraction_bug_vs_fix(completion: str, v1_expected: str, v2_expected: str) -> None:
    # V1 leaves \left / ^\circ / \! in place (the preserved bug); V2 strips them as intended.
    assert _v1(completion) == v1_expected
    assert _v2(completion) == v2_expected


# ---------------------------------------------------------------------------
# GSM8KReasoning: boxed-answer prompt (zero-shot); extraction falls back to the #### line
# ---------------------------------------------------------------------------

_GSM8K_ROW: dict[str, Any] = {"question": "What is 2 + 2?", "answer": "Two plus two is four.\n#### 4"}


def test_gsm8k_reasoning_prompt() -> None:
    benchmark = gsm8k_reasoning(dataset=DatasetStub({"test": [_GSM8K_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["main"])
    assert sample.messages == [
        Message(role=Role.USER, content=_GSM8K_REASONING_QUERY_TEMPLATE.format(question="What is 2 + 2?"))
    ]
    assert sample.ground_truth == "4"  # the #### line of the gold answer, commas stripped
    assert sample.possible_completions is None  # free-form generation


@pytest.mark.parametrize(
    "completion, expected",
    [
        (r"So \boxed{42}", "42"),  # boxed wins
        (r"\boxed{1,024}", "1024"),  # boxed, commas stripped
        ("The final answer is 12.\n#### 42", "42"),  # no box -> #### fallback (not the inline 12)
        ("#### 1,000", "1000"),  # #### fallback, commas stripped
        ("no boxed and no hash line", "[invalid]"),
    ],
)
def test_gsm8k_reasoning_extraction(completion: str, expected: str) -> None:
    assert _gsm8k_reasoning_extractor(completion) == expected
