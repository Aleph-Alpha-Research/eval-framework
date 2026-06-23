"""Tests for MMLU_DE formatter hashes and offline prompt construction."""

import pytest

from eval_framework.tasks.benchmarks.mmlu_de import MMLU_DE
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.tasks.benchmarks.utils import (
    ExpectedPrompt,
    OfflinePromptCase,
    get_task_names_for_module,
    run_formatter_hash_test,
    run_offline_prompt_case,
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

# Expected prompts (messages, ground truth, completions). The rendered string is the
# formatter's concern and is covered in tests/tests_template_formatting/test_formatter_eval.py.
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

_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=_INTRO + _EVAL_QUESTION),
        Message(role=Role.ASSISTANT, content=_CUE),
    ],
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
    ground_truth=_GROUND_TRUTH,
    completions=_COMPLETIONS,
)

# Adding a new task is just another entry in this list (see OfflinePromptCase).
CASES = [
    OfflinePromptCase(
        task_cls=MMLU_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        zeroshot=_ZEROSHOT,
        oneshot=_FEWSHOT,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[case.test_id for case in CASES])
def test_offline_prompt_assembly(case: OfflinePromptCase) -> None:
    run_offline_prompt_case(case)
