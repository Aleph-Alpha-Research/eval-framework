"""Tests for MMLU_DE formatter hashes and offline prompt construction."""

import pytest

from eval_framework.tasks.benchmarks.mmlu_de import MMLU_DE
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.tasks.benchmarks.utils import (
    ExpectedPrompt,
    assert_offline_oneshot_prompt,
    assert_offline_zeroshot_prompt,
    get_task_names_for_module,
    run_formatter_hash_test,
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
# Offline prompt tests (patched HF load; production task API otherwise)
# ---------------------------------------------------------------------------

_SUBJECT = "abstract_algebra"  # rendered as "Abstrakte Algebra" in the intro line

# Dummy offline rows (obvious test content, fictional data).
_EVAL_ROW: dict[str] = {
    "answer": 1,  # -> ground truth " B"
    "question_de": (
        "Aussage 1 | Papageien können keine Farben sehen und leben nur in Schwarz-Weiß. "
        "Aussage 2 | Graupapageien sind berühmt dafür, dass sie niemals Menschenstimmen nachahmen."
    ),
    "choices_de": ["Wahr, Wahr", "Falsch, Falsch", "Wahr, Falsch", "Falsch, Wahr"],
}

_FEWSHOT_ROW: dict[str] = {
    "answer": 2,  # -> few-shot answer "Antwort: C"
    "question_de": (
        "Aussage 1 | Tiger sind die größte lebende Katzenart und haben ein Streifenmuster wie ein Fingerabdruck. "
        "Aussage 2 | Ein Test-Tiger in diesem Offline-Benchmark springt zuverlässig 50 Meter weit, wenn niemand hinschaut."
    ),
    "choices_de": ["Wahr, Wahr", "Falsch, Falsch", "Wahr, Falsch", "Falsch, Wahr"],
}

# Expected prompts (messages, flat concat, ground truth, completions).
_INTRO = """\
Die folgenden sind Multiple Choice Fragen (mit Antworten) über Abstrakte Algebra.

"""

_EVAL_QUESTION = """\
Frage: Aussage 1 | Papageien können keine Farben sehen und leben nur in Schwarz-Weiß. Aussage 2 | Graupapageien sind berühmt dafür, dass sie niemals Menschenstimmen nachahmen.
A. Wahr, Wahr
B. Falsch, Falsch
C. Wahr, Falsch
D. Falsch, Wahr
"""

_FEWSHOT_QUESTION = """\
Frage: Aussage 1 | Tiger sind die größte lebende Katzenart und haben ein Streifenmuster wie ein Fingerabdruck. Aussage 2 | Ein Test-Tiger in diesem Offline-Benchmark springt zuverlässig 50 Meter weit, wenn niemand hinschaut.
A. Wahr, Wahr
B. Falsch, Falsch
C. Wahr, Falsch
D. Falsch, Wahr
"""

_CUE = "Antwort:"
_FEWSHOT_ANSWER = "Antwort: C"
_GROUND_TRUTH = " B"
_COMPLETIONS = [" A", " B", " C", " D"]

# _INTRO + _EVAL_QUESTION + _CUE
_EXPECTED_CONCAT_0SHOT = """\
Die folgenden sind Multiple Choice Fragen (mit Antworten) über Abstrakte Algebra.

Frage: Aussage 1 | Papageien können keine Farben sehen und leben nur in Schwarz-Weiß. Aussage 2 | Graupapageien sind berühmt dafür, dass sie niemals Menschenstimmen nachahmen.
A. Wahr, Wahr
B. Falsch, Falsch
C. Wahr, Falsch
D. Falsch, Wahr
Antwort:"""

# _INTRO + _FEWSHOT_QUESTION + _FEWSHOT_ANSWER + _EXAMPLE_SEPARATOR + _EVAL_QUESTION + _CUE
_EXPECTED_CONCAT_1SHOT = """\
Die folgenden sind Multiple Choice Fragen (mit Antworten) über Abstrakte Algebra.

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

_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=_INTRO + _EVAL_QUESTION),
        Message(role=Role.ASSISTANT, content=_CUE),
    ],
    concat=_EXPECTED_CONCAT_0SHOT,
    ground_truth=_GROUND_TRUTH,
    completions=_COMPLETIONS,
)

_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=_INTRO + _FEWSHOT_QUESTION),
        Message(role=Role.ASSISTANT, content=_FEWSHOT_ANSWER),
        Message(role=Role.USER, content=_EVAL_QUESTION),
        Message(role=Role.ASSISTANT, content=_CUE),
    ],
    concat=_EXPECTED_CONCAT_1SHOT,
    ground_truth=_GROUND_TRUTH,
    completions=_COMPLETIONS,
)


def test_mmlu_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        MMLU_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        MMLU_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_FEWSHOT,
    )
