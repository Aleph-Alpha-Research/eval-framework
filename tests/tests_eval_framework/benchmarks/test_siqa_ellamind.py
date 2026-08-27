"""Tests for the German Social IQa (EllaMind) tasks.

- formatter hash test for every SIQA variant
- offline test that the reader (item -> ChoiceFields) and the chosen styler produce the expected
  prompt content. Message assembly (roles / fewshot / cue placement) is generic and covered in
  ``test_composed_benchmark``.
"""

from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.benchmarks.siqa_ellamind import (
    SIQA_ELLAMIND_BPB_STYLER,
    SIQA_ELLAMIND_CLOZE_STYLER,
    SIQA_ELLAMIND_MC_STYLER,
    SiqaReader,
)
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_siqa_ellamind_tasks
from eval_framework.tasks.task_style import TaskStyler
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding siqa_ellamind tasks
_siqa_ellamind_registry = Registry()
register_siqa_ellamind_tasks(registry=_siqa_ellamind_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _siqa_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_siqa_ellamind_registry)


# ---------------------------------------------------------------------------
# Offline test: reader + chosen styler produce the expected prompt content (no Eval, no dataset)
# ---------------------------------------------------------------------------

# A fictional row in the SIQA format (NOT a real dataset example). The shown question is the context
# followed by the question; choices are shuffled deterministically (seed: question + correct_answer).
_EVAL_ROW: dict[str, Any] = {
    "context": "Alex hat den ganzen Tag im Garten gearbeitet.",
    "question": "Wie fühlt sich Alex danach?",
    "correct_answer": "Erschöpft",
    "easy_distractors": ["Gelangweilt", "Neugierig", "Hungrig"],
    "hard_distractors": ["Zufrieden", "Entspannt", "Stolz"],
}

_QUESTION = "Frage: Alex hat den ganzen Tag im Garten gearbeitet. Wie fühlt sich Alex danach?"


@dataclass(frozen=True)
class _ExpectedPrompt:
    instruction: str
    cue: str
    ground_truth: str
    completions: list[str]


_MC_EASY = _ExpectedPrompt(
    instruction=f"{_QUESTION}\nA. Hungrig\nB. Neugierig\nC. Gelangweilt\nD. Erschöpft\n",
    cue="Antwort:",
    ground_truth=" D",
    completions=[" A", " B", " C", " D"],
)
_MC_HARD = _ExpectedPrompt(
    instruction=f"{_QUESTION}\nA. Stolz\nB. Entspannt\nC. Zufrieden\nD. Erschöpft\n",
    cue="Antwort:",
    ground_truth=" D",
    completions=[" A", " B", " C", " D"],
)
# Cloze/BPB show no options, so the prompt text is identical; only the scored completions differ.
_CLOZE_EASY = _ExpectedPrompt(
    instruction=f"{_QUESTION}\n",
    cue="Antwort:",
    ground_truth=" Erschöpft",
    completions=[" Hungrig", " Neugierig", " Gelangweilt", " Erschöpft"],
)
_CLOZE_HARD = _ExpectedPrompt(
    instruction=f"{_QUESTION}\n",
    cue="Antwort:",
    ground_truth=" Erschöpft",
    completions=[" Stolz", " Entspannt", " Zufrieden", " Erschöpft"],
)
_BPB = _ExpectedPrompt(
    instruction=f"{_QUESTION}\n",
    cue="Antwort:",
    ground_truth=" Erschöpft",
    completions=[" Erschöpft"],  # BPB scores only the gold continuation
)


@pytest.mark.parametrize(
    "reader, styler, expected",
    [
        pytest.param(SiqaReader("easy"), SIQA_ELLAMIND_MC_STYLER, _MC_EASY, id="mc_easy"),
        pytest.param(SiqaReader("hard"), SIQA_ELLAMIND_MC_STYLER, _MC_HARD, id="mc_hard"),
        pytest.param(SiqaReader("easy"), SIQA_ELLAMIND_CLOZE_STYLER, _CLOZE_EASY, id="cloze_easy"),
        pytest.param(SiqaReader("hard"), SIQA_ELLAMIND_CLOZE_STYLER, _CLOZE_HARD, id="cloze_hard"),
        pytest.param(SiqaReader("easy"), SIQA_ELLAMIND_BPB_STYLER, _BPB, id="bpb"),
    ],
)
def test_siqa_prompt_content(reader: SiqaReader, styler: TaskStyler, expected: _ExpectedPrompt) -> None:
    fields = reader.read(_EVAL_ROW)
    assert styler.get_instruction_text(fields.raw_question, fields.choices) == expected.instruction
    assert styler.get_cue_text() == expected.cue
    assert styler.get_ground_truth(fields.choices, fields.correct_index) == expected.ground_truth
    assert styler.get_possible_completions(fields.choices, fields.correct_index) == expected.completions
