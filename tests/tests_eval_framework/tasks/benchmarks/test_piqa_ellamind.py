"""Tests for the German PIQA (EllaMind) tasks.

- formatter hash test for every PIQA variant
- offline test that the reader (item -> ChoiceFields) and the chosen styler produce the expected
  prompt content. Message assembly (roles / fewshot / cue placement) is generic and covered in
  ``test_composed_benchmark``.
"""

from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.choices import ChoiceReader
from eval_framework.tasks.benchmarks.piqa_ellamind import (
    PIQA_ELLAMIND_BPB_STYLER,
    PIQA_ELLAMIND_CLOZE_STYLER,
    PIQA_ELLAMIND_MC_STYLER,
    piqa_reader,
)
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_piqa_ellamind_tasks
from eval_framework.tasks.task_style import TaskStyler
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, NoStripConcatFormatter
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding piqa_ellamind tasks
_piqa_ellamind_registry = Registry()
register_piqa_ellamind_tasks(registry=_piqa_ellamind_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _piqa_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_piqa_ellamind_registry)


# ---------------------------------------------------------------------------
# Offline test: reader + chosen styler produce the expected prompt content (no Eval, no dataset)
# ---------------------------------------------------------------------------

# A fictional row in the PIQA format (NOT a real dataset example). Choices are shuffled
# deterministically (seed: goal + correct_solution), which places the correct answer at index 0.
_EVAL_ROW: dict[str, Any] = {
    "goal": "Martin möchte einen Nagel in die Wand schlagen.",
    "correct_solution": "Er verwendet einen Hammer.",
    "easy_distractor": "Er verwendet eine Schere.",
    "hard_distractor": "Er verwendet eine Zange.",
}


@dataclass(frozen=True)
class _ExpectedPrompt:
    instruction: str
    cue: str
    ground_truth: str
    completions: list[str]


_MC_EASY = _ExpectedPrompt(
    instruction="Ziel: Martin möchte einen Nagel in die Wand schlagen.\n"
    "A. Er verwendet einen Hammer.\nB. Er verwendet eine Schere.\n",
    cue="Antwort:",
    ground_truth=" A",
    completions=[" A", " B"],
)
_MC_HARD = _ExpectedPrompt(
    instruction="Ziel: Martin möchte einen Nagel in die Wand schlagen.\n"
    "A. Er verwendet einen Hammer.\nB. Er verwendet eine Zange.\n",
    cue="Antwort:",
    ground_truth=" A",
    completions=[" A", " B"],
)
# Cloze/BPB show no options, so the prompt text is identical; only the scored completions differ.
_CLOZE_EASY = _ExpectedPrompt(
    instruction="Ziel: Martin möchte einen Nagel in die Wand schlagen.\n",
    cue="Antwort:",
    ground_truth=" Er verwendet einen Hammer.",
    completions=[" Er verwendet einen Hammer.", " Er verwendet eine Schere."],
)
_CLOZE_HARD = _ExpectedPrompt(
    instruction="Ziel: Martin möchte einen Nagel in die Wand schlagen.\n",
    cue="Antwort:",
    ground_truth=" Er verwendet einen Hammer.",
    completions=[" Er verwendet einen Hammer.", " Er verwendet eine Zange."],
)
_BPB = _ExpectedPrompt(
    instruction="Ziel: Martin möchte einen Nagel in die Wand schlagen.\n",
    cue="Antwort:",
    ground_truth=" Er verwendet einen Hammer.",
    completions=[" Er verwendet einen Hammer."],  # BPB scores only the gold continuation
)


@pytest.mark.parametrize(
    "reader, styler, expected",
    [
        pytest.param(piqa_reader("easy"), PIQA_ELLAMIND_MC_STYLER, _MC_EASY, id="mc_easy"),
        pytest.param(piqa_reader("hard"), PIQA_ELLAMIND_MC_STYLER, _MC_HARD, id="mc_hard"),
        pytest.param(piqa_reader("easy"), PIQA_ELLAMIND_CLOZE_STYLER, _CLOZE_EASY, id="cloze_easy"),
        pytest.param(piqa_reader("hard"), PIQA_ELLAMIND_CLOZE_STYLER, _CLOZE_HARD, id="cloze_hard"),
        pytest.param(piqa_reader("easy"), PIQA_ELLAMIND_BPB_STYLER, _BPB, id="bpb"),
    ],
)
def test_piqa_prompt_content(reader: ChoiceReader, styler: TaskStyler, expected: _ExpectedPrompt) -> None:
    fields = reader.read(_EVAL_ROW)
    assert styler.get_instruction_text(fields.raw_question, fields.choices) == expected.instruction
    assert styler.get_cue_text() == expected.cue
    assert styler.get_ground_truth(fields.choices, fields.correct_index) == expected.ground_truth
    assert styler.get_possible_completions(fields.choices, fields.correct_index) == expected.completions
