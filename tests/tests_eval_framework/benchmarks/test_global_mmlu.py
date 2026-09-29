"""Specification of the composed Global-MMLU tasks.

Global-MMLU is MMLU per language: each subject is a ``(language, subject)`` pair, and the whole prompt —
preamble, the "Question"/"Answer" labels, and the subject name — is rendered in that language. The spec
tests build the real benchmark over one fictional row for two languages and assert the localized sample;
``test_formatter_hash`` pins both variants against the real HuggingFace data.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.global_mmlu import GLOBAL_MMLU_BENCHMARKS, global_mmlu
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
@pytest.mark.parametrize("benchmark", GLOBAL_MMLU_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over one fictional row, assert the localized sample
# ---------------------------------------------------------------------------

# Fictional Global-MMLU row (NOT a real example). ``subject`` (the column filter) is inert here because the
# stub injects a single row directly.
_EVAL_ROW: dict[str, Any] = {
    "question": "  What is 2+2?  ",  # surrounding whitespace is stripped
    "option_a": "3",
    "option_b": "4",
    "option_c": "5",
    "option_d": "6",
    "answer": "B",
    "subject": "abstract_algebra",
}


@pytest.mark.parametrize(
    "subject_label, preamble, question_word, answer_word",
    [
        (
            "('de', 'abstract_algebra')",
            "Die folgenden sind Multiple-Choice-Fragen (mit Antworten) über Abstrakte Algebra.",
            "Frage",
            "Antwort",
        ),
        (
            "('fr', 'abstract_algebra')",
            "Les questions suivantes sont des questions à choix multiples (avec réponses) sur Algèbre Abstraite.",
            "Question",
            "Réponse",
        ),
    ],
)
def test_global_mmlu_prompt_is_localized(
    subject_label: str, preamble: str, question_word: str, answer_word: str
) -> None:
    benchmark = global_mmlu(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=[subject_label])
    assert sample.messages == [
        Message(role=Role.USER, content=f"{preamble}\n\n{question_word}: What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\n"),
        Message(role=Role.ASSISTANT, content=f"{answer_word}:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [" A", " B", " C", " D"]
