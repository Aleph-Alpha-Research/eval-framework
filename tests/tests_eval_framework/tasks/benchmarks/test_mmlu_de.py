"""Tests for MMLU_DE formatter hashes and offline prompt construction."""

from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from eval_framework.tasks.base import Sample
from eval_framework.tasks.benchmarks.mmlu_de import MMLU_DE
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.tasks.benchmarks.utils import get_task_names_for_module, run_formatter_hash_test

# ---------------------------------------------------------------------------
# Formatter hash tests (Hugging Face)
# ---------------------------------------------------------------------------


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", get_task_names_for_module("mmlu_de"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls)


# ---------------------------------------------------------------------------
# Offline prompt tests (patched HF load; production task API otherwise)
# ---------------------------------------------------------------------------

_SUBJECT = "abstract_algebra"

# Dummy offline rows (obvious test content, fictional data).
_EVAL_ROW = {
    "answer": 1,
    "question_de": (
        "Aussage 1 | Papageien können keine Farben sehen und leben nur in Schwarz-Weiß. "
        "Aussage 2 | Graupapageien sind berühmt dafür, dass sie niemals Menschenstimmen nachahmen."
    ),
    "choices_de": ["Wahr, Wahr", "Falsch, Falsch", "Wahr, Falsch", "Falsch, Wahr"],
}

_FEWSHOT_ROW = {
    "answer": 2,
    "question_de": (
        "Aussage 1 | Tiger sind die größte lebende Katzenart und haben ein Streifenmuster wie ein Fingerabdruck. "
        "Aussage 2 | Ein Test-Tiger in diesem Offline-Benchmark springt zuverlässig 50 Meter weit, wenn niemand hinschaut."
    ),
    "choices_de": ["Wahr, Wahr", "Falsch, Falsch", "Wahr, Falsch", "Falsch, Wahr"],
}


def _fictional_dataset(task: MMLU_DE) -> DatasetDict:
    return DatasetDict(
        {
            task.SAMPLE_SPLIT: Dataset.from_list([_EVAL_ROW]),
            task.FEWSHOT_SPLIT: Dataset.from_list([_FEWSHOT_ROW]),
        }
    )


def _first_sample(*, num_fewshot: int) -> Sample:
    """Same entry points as production: ``with_overwrite`` + ``iterate_samples``."""
    task = MMLU_DE.with_overwrite(
        num_fewshot=num_fewshot,
        custom_subjects=[_SUBJECT],
        custom_hf_revision=None,
    )
    with patch.object(task, "_load_hf_dataset", return_value=_fictional_dataset(task)):
        return next(iter(task.iterate_samples(1)))


class TestMMLU_DEWithHardcodedData:
    """Offline prompt, ground-truth, and completion formatting for MMLU_DE."""

    def test_0shot_prompt(self) -> None:
        sample = _first_sample(num_fewshot=0)

        assert sample.messages == _EXPECTED_0SHOT_EVAL_MESSAGES
        assert sample.ground_truth == " B"
        assert sample.possible_completions == [" A", " B", " C", " D"]

    def test_1shot_prompt(self) -> None:
        sample = _first_sample(num_fewshot=1)

        assert sample.messages == _EXPECTED_1SHOT_MESSAGES
        assert sample.ground_truth == " B"
        assert sample.possible_completions == [" A", " B", " C", " D"]


INTRO_PROMPT = """Die folgenden sind Multiple Choice Fragen (mit Antworten) über Abstrakte Algebra.

"""

EVAL_QUESTION_PROMPT = (
    "Frage: Aussage 1 | Papageien können keine Farben sehen und leben nur in Schwarz-Weiß. "
    "Aussage 2 | Graupapageien sind berühmt dafür, dass sie niemals Menschenstimmen nachahmen.\n"
    "A. Wahr, Wahr\n"
    "B. Falsch, Falsch\n"
    "C. Wahr, Falsch\n"
    "D. Falsch, Wahr\n"
)

FEWSHOT_QUESTION_PROMPT = (
    "Frage: Aussage 1 | Tiger sind die größte lebende Katzenart und haben ein Streifenmuster wie ein Fingerabdruck. "
    "Aussage 2 | Ein Test-Tiger in diesem Offline-Benchmark springt zuverlässig 50 Meter weit, wenn niemand hinschaut.\n"
    "A. Wahr, Wahr\n"
    "B. Falsch, Falsch\n"
    "C. Wahr, Falsch\n"
    "D. Falsch, Wahr\n"
)

ANSWER_CUE = "Antwort:"
FEWSHOT_ANSWER = "Antwort: C"

_EXPECTED_0SHOT_EVAL_MESSAGES = [
    Message(role=Role.USER, content=INTRO_PROMPT + EVAL_QUESTION_PROMPT),
    Message(role=Role.ASSISTANT, content=ANSWER_CUE),
]

_EXPECTED_1SHOT_MESSAGES = [
    Message(role=Role.USER, content=INTRO_PROMPT + FEWSHOT_QUESTION_PROMPT),
    Message(role=Role.ASSISTANT, content=FEWSHOT_ANSWER),
    Message(role=Role.USER, content=EVAL_QUESTION_PROMPT),
    Message(role=Role.ASSISTANT, content=ANSWER_CUE),
]

EXPECTED_FULL_CONCAT_OUTPUT = """Die folgenden sind Multiple Choice Fragen (mit Antworten) über Abstrakte Algebra.

Frage: Aussage 1 | Tiger sind die größte lebende Katzenart und haben ein Streifenmuster wie ein Fingerabdruck. Aussage 2 | Ein Test-Tiger in diesem Offline-Benchmark springt zuverlässig 50 Meter weit, wenn niemand hinschaut.
A. Wahr, Wahr
B. Falsch, Falsch
C. Wahr, Falsch
D. Falsch, Wahr
Antwort: C

Frage: Aussage 1 | Papageien können keine Farben sehen und leben nur in Schwarz-Weiß. Aussage 2 | Graupapageien sind berühmt dafür, dass sie niemals Menschenstimmen nachahmen.
A. Wahr, Wahr
B. Falsch, Falsch
C. Wahr, Falsch
D. Falsch, Wahr
Antwort:"""


concat_formater = ConcatFormatter()
concat_formatted = concat_formater.format(_EXPECTED_1SHOT_MESSAGES, output_mode="string")
assert concat_formatted == EXPECTED_FULL_CONCAT_OUTPUT
