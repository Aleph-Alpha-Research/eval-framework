"""Tests for the German Hendrycks Math (EllaMind) tasks.

Tests:
- formatter hash test for every Hendrycks Math variant
- offline prompt assembly tests (generative MATHMinerva format and BPB format)
"""

from typing import Any

import pytest

import eval_framework.tasks.benchmarks.hendrycks_math_ellamind as hendrycks_math_ellamind
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_hendrycks_math_ellamind_tasks
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    NoStripConcatFormatter,
    Role,
)
from tests.tests_eval_framework.tasks.benchmarks.utils import (
    ExpectedPrompt,
    assert_offline_oneshot_prompt,
    assert_offline_zeroshot_prompt,
    run_formatter_hash_test,
)

# Registry for this test suite only holding hendrycks_math_ellamind tasks
_hendrycks_math_ellamind_registry = Registry()
register_hendrycks_math_ellamind_tasks(registry=_hendrycks_math_ellamind_registry)

# ---------------------------------------------------------------------------
# Formatter hash tests (Hugging Face)
# ---------------------------------------------------------------------------


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _hendrycks_math_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_hendrycks_math_ellamind_registry)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "deu"

# Fictional rows following the Hendrycks Math format. NOT real examples from the dataset.
_EVAL_ROW: dict[str, Any] = {
    "problem": "Was ist 2 + 2?",
    "solution": "Die Antwort ist \\boxed{4}.",
    "extracted_answer": "4",
}

_FEWSHOT_ROW: dict[str, Any] = {
    "problem": "Was ist 3 + 3?",
    "solution": "Die Antwort ist \\boxed{6}.",
    "extracted_answer": "6",
}

# Expected prompts (messages, flat concat, ground truth, completions).
# --- MATHMinervaDE_OLMES ---
_OLMES_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Aufgabe:\nWas ist 2 + 2?\n\n"),
        Message(role=Role.ASSISTANT, content="Lösung:"),
    ],
    concat="""\
Aufgabe:
Was ist 2 + 2?

Lösung:""",
    ground_truth="4",
    completions=None,
)

_OLMES_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Aufgabe:\nWas ist 3 + 3?\n\n"),
        Message(
            role=Role.ASSISTANT,
            content="Lösung: Die Antwort ist \\boxed{6}.\nFinale Antwort: Die finale Antwort lautet $6$. Ich hoffe, die Antwort ist korrekt.",
        ),
        Message(role=Role.USER, content="Aufgabe:\nWas ist 2 + 2?\n\n"),
        Message(role=Role.ASSISTANT, content="Lösung:"),
    ],
    concat="""\
Aufgabe:
Was ist 3 + 3?

Lösung: Die Antwort ist \\boxed{6}.
Finale Antwort: Die finale Antwort lautet $6$. Ich hoffe, die Antwort ist korrekt.

Aufgabe:
Was ist 2 + 2?

Lösung:""",
    ground_truth=_OLMES_ZEROSHOT.ground_truth,
    completions=_OLMES_ZEROSHOT.completions,
)

# --- MATHMinervaDE_BPB_OLMES ---
_BPB_GROUND_TRUTH = (
    " Die Antwort ist \\boxed{4}.\nFinale Antwort: Die finale Antwort lautet $4$. Ich hoffe, die Antwort ist korrekt."
)

_BPB_ZEROSHOT = ExpectedPrompt(
    messages=_OLMES_ZEROSHOT.messages,
    concat=_OLMES_ZEROSHOT.concat,
    ground_truth=_BPB_GROUND_TRUTH,
    completions=[_BPB_GROUND_TRUTH],
)

_BPB_FEWSHOT = ExpectedPrompt(
    messages=_OLMES_FEWSHOT.messages,
    concat=_OLMES_FEWSHOT.concat,
    ground_truth=_BPB_GROUND_TRUTH,
    completions=[_BPB_GROUND_TRUTH],
)


# --- TESTS ---
def test_mathminervade_olmes_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        hendrycks_math_ellamind.MATHMinervaDE_OLMES,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_OLMES_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        hendrycks_math_ellamind.MATHMinervaDE_OLMES,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_OLMES_FEWSHOT,
    )


def test_mathminervade_bpb_olmes_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        hendrycks_math_ellamind.MATHMinervaDE_BPB_OLMES,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        hendrycks_math_ellamind.MATHMinervaDE_BPB_OLMES,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )
