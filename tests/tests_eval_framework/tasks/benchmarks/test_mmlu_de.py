"""Tests for MMLU_DE formatter hashes and offline prompt construction."""

import pytest

from eval_framework.tasks.base import Sample
from eval_framework.tasks.benchmarks.mmlu_de import MMLU_DE
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter
from tests.tests_eval_framework.tasks.benchmarks.utils import get_task_names_for_module, run_formatter_hash_test
from tests.tests_eval_framework.tasks.offline_prompt_test_utils import (
    assert_loglikelihood_targets_consistent,
    load_offline_task,
    prompt_string,
)

# ---------------------------------------------------------------------------
# Formatter hash tests (Hugging Face)
# ---------------------------------------------------------------------------


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", get_task_names_for_module("mmlu_de"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls)


# ---------------------------------------------------------------------------
# Offline prompt tests
# ---------------------------------------------------------------------------

_SUBJECT = "abstract_algebra"

# Fictional mock items — NOT real examples from LeoLM/MMLU_de.
# Field names and shapes match the HF dataset; content is invented for offline tests.
_EVAL_ROW = {
    "question": "How many elements does the Klein four-group V4 have?",
    "choices": ["2", "3", "4", "8"],
    "answer": 2,
    "question_de": "Wie viele Elemente hat die Kleinsche Vierergruppe V₄?",
    "choices_de": ["2", "3", "4", "8"],
    "answer_de": (
        '{"question": "Wie viele Elemente hat die Kleinsche Vierergruppe V₄?", "A": "2", "B": "3", "C": "4", "D": "8"}'
    ),
}

_FEWSHOT_ROW = {
    "question": "Is every subgroup of an abelian group normal?",
    "choices": ["No, never", "Yes, always", "Only if the group is cyclic", "Only in the finite case"],
    "answer": 1,
    "question_de": "Ist jede Untergruppe einer abelschen Gruppe normal?",
    "choices_de": ["Nein, niemals", "Ja, immer", "Nur wenn die Gruppe zyklisch ist", "Nur im endlichen Fall"],
    "answer_de": (
        '{"question": "Ist jede Untergruppe einer abelschen Gruppe normal?", '
        '"A": "Nein, niemals", "B": "Ja, immer", '
        '"C": "Nur wenn die Gruppe zyklisch ist", "D": "Nur im endlichen Fall"}'
    ),
}


def _load_mmlu_de(*, num_fewshot: int) -> MMLU_DE:
    return load_offline_task(
        MMLU_DE,
        subject=_SUBJECT,
        num_fewshot=num_fewshot,
        eval_row=_EVAL_ROW,
        fewshot_pool=[_FEWSHOT_ROW],
    )


def _eval_sample(task: MMLU_DE) -> Sample:
    item = {**task.dataset[task.SAMPLE_SPLIT][0], "subject": _SUBJECT}
    return task._create_samples(item, index=0, subject=_SUBJECT)[0]


class TestMMLU_DEWithHardcodedData:
    """Offline prompt, ground-truth, and completion formatting for MMLU_DE."""

    def test_0shot_prompt(self) -> None:
        task = _load_mmlu_de(num_fewshot=0)
        sample = _eval_sample(task)

        assert prompt_string(sample) == _EXPECTED_0SHOT_PROMPT
        assert sample.ground_truth == " C"
        assert sample.possible_completions == [" A", " B", " C", " D"]
        assert_loglikelihood_targets_consistent(sample)

    def test_1shot_prompt(self) -> None:
        task = _load_mmlu_de(num_fewshot=1)
        sample = _eval_sample(task)

        assert prompt_string(sample) == _EXPECTED_1SHOT_PROMPT
        assert sample.ground_truth == " C"
        assert sample.possible_completions == [" A", " B", " C", " D"]
        assert_loglikelihood_targets_consistent(sample)


# Golden prompts for the fictional rows above (from production MMLU_DE message builders).
_EXPECTED_0SHOT_PROMPT = """\
[user] Die folgenden sind Multiple Choice Fragen (mit Antworten) über Abstrakte Algebra.

Frage: Wie viele Elemente hat die Kleinsche Vierergruppe V₄?
A. 2
B. 3
C. 4
D. 8

[assistant] Antwort:"""

_EXPECTED_1SHOT_PROMPT = """\
[user] Die folgenden sind Multiple Choice Fragen (mit Antworten) über Abstrakte Algebra.

Frage: Ist jede Untergruppe einer abelschen Gruppe normal?
A. Nein, niemals
B. Ja, immer
C. Nur wenn die Gruppe zyklisch ist
D. Nur im endlichen Fall

[assistant] Antwort: B
[user] Frage: Wie viele Elemente hat die Kleinsche Vierergruppe V₄?
A. 2
B. 3
C. 4
D. 8

[assistant] Antwort:"""
