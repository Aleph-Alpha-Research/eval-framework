"""Tests for the German CSQA (EllaMind) tasks.

- formatter hash test for every CSQA variant
- offline test that the reader (item -> ChoiceFields) and the chosen styler produce the expected
  prompt content. Message assembly (roles / fewshot / cue placement) is generic and covered in
  ``test_composed_benchmark``.
- reader test that distractor shuffling is deterministic and uses the selected distractor set.
"""

from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.tasks.benchmarks.csqa_ellamind import (
    CSQA_ELLAMIND_BPB_STYLER,
    CSQA_ELLAMIND_CLOZE_STYLER,
    CSQA_ELLAMIND_MC_STYLER,
    CsqaReader,
)
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_csqa_ellamind_tasks
from eval_framework.tasks.task_style import TaskStyler
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding csqa_ellamind tasks
_csqa_ellamind_registry = Registry()
register_csqa_ellamind_tasks(registry=_csqa_ellamind_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _csqa_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_csqa_ellamind_registry)


# ---------------------------------------------------------------------------
# Offline test: reader + chosen styler produce the expected prompt content (no Eval, no dataset)
# ---------------------------------------------------------------------------

# A fictional row in the CSQA format (NOT a real dataset example). Choices are shuffled
# deterministically (seed: question + correct_answer), which places the correct answer at index 0.
_EVAL_ROW: dict[str, Any] = {
    "question": "Wo bewahrt man frische Milch am besten auf?",
    "correct_answer": "Im Kühlschrank",
    "easy_distractors": ["Auf dem Mond", "In einem Schuh", "Im Vulkan"],
    "hard_distractors": ["In der Speisekammer", "Auf der Fensterbank", "Im Keller"],
}


@dataclass(frozen=True)
class _ExpectedPrompt:
    instruction: str
    cue: str
    ground_truth: str
    completions: list[str]


_MC_EASY = _ExpectedPrompt(
    instruction="Frage: Wo bewahrt man frische Milch am besten auf?\n"
    "A. Im Kühlschrank\nB. Auf dem Mond\nC. In einem Schuh\nD. Im Vulkan\n",
    cue="Antwort:",
    ground_truth=" A",
    completions=[" A", " B", " C", " D"],
)
_MC_HARD = _ExpectedPrompt(
    instruction="Frage: Wo bewahrt man frische Milch am besten auf?\n"
    "A. Im Kühlschrank\nB. In der Speisekammer\nC. Auf der Fensterbank\nD. Im Keller\n",
    cue="Antwort:",
    ground_truth=" A",
    completions=[" A", " B", " C", " D"],
)
# Cloze/BPB show no options, so the prompt text is identical; only the scored completions differ.
_CLOZE_EASY = _ExpectedPrompt(
    instruction="Frage: Wo bewahrt man frische Milch am besten auf?\n",
    cue="Antwort:",
    ground_truth=" Im Kühlschrank",
    completions=[" Im Kühlschrank", " Auf dem Mond", " In einem Schuh", " Im Vulkan"],
)
_CLOZE_HARD = _ExpectedPrompt(
    instruction="Frage: Wo bewahrt man frische Milch am besten auf?\n",
    cue="Antwort:",
    ground_truth=" Im Kühlschrank",
    completions=[" Im Kühlschrank", " In der Speisekammer", " Auf der Fensterbank", " Im Keller"],
)
_BPB = _ExpectedPrompt(
    instruction="Frage: Wo bewahrt man frische Milch am besten auf?\n",
    cue="Antwort:",
    ground_truth=" Im Kühlschrank",
    completions=[" Im Kühlschrank"],  # BPB scores only the gold continuation
)


@pytest.mark.parametrize(
    "reader, styler, expected",
    [
        pytest.param(CsqaReader("easy"), CSQA_ELLAMIND_MC_STYLER, _MC_EASY, id="mc_easy"),
        pytest.param(CsqaReader("hard"), CSQA_ELLAMIND_MC_STYLER, _MC_HARD, id="mc_hard"),
        pytest.param(CsqaReader("easy"), CSQA_ELLAMIND_CLOZE_STYLER, _CLOZE_EASY, id="cloze_easy"),
        pytest.param(CsqaReader("hard"), CSQA_ELLAMIND_CLOZE_STYLER, _CLOZE_HARD, id="cloze_hard"),
        pytest.param(CsqaReader("easy"), CSQA_ELLAMIND_BPB_STYLER, _BPB, id="bpb"),
    ],
)
def test_csqa_prompt_content(reader: CsqaReader, styler: TaskStyler, expected: _ExpectedPrompt) -> None:
    fields = reader.read(_EVAL_ROW)
    assert styler.get_instruction_text(fields.raw_question, fields.choices) == expected.instruction
    assert styler.get_cue_text() == expected.cue
    assert styler.get_ground_truth(fields.choices, fields.correct_index) == expected.ground_truth
    assert styler.get_possible_completions(fields.choices, fields.correct_index) == expected.completions


def test_csqa_reader_is_deterministic_and_uses_the_selected_distractor_set() -> None:
    item: dict[str, Any] = {
        "question": "Was essen Pandas am liebsten?",
        "correct_answer": "Bambus",
        "easy_distractors": ["Pizza", "Eis", "Schokolade"],
        "hard_distractors": ["Blätter", "Gräser", "Kräuter"],
    }
    easy = CsqaReader("easy").read(item)
    hard = CsqaReader("hard").read(item)

    # Deterministic for identical input, and the correct index points to the correct answer
    assert CsqaReader("easy").read(item) == easy
    assert easy.choices[easy.correct_index] == "Bambus"
    assert hard.choices[hard.correct_index] == "Bambus"
    # Easy and hard draw from different distractor pools
    assert set(easy.choices) == {"Bambus", "Pizza", "Eis", "Schokolade"}
    assert set(hard.choices) == {"Bambus", "Blätter", "Gräser", "Kräuter"}
