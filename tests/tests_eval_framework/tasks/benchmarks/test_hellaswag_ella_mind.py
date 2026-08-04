"""Tests for the German HellaSwag (EllaMind) tasks.

Tests:
- formatter hash test for every HellaSwag variant
- offline prompt assembly tests

HellaSwag is a sentence-completion task: the prompt is the partial sentence
``"{activity}: {context}"`` with no question/answer cue, and the model scores full sentence
endings. The easy/hard/BPB variants therefore share the same prompt and differ only in the
scored completions.
"""

from typing import Any

import pytest

import eval_framework.tasks.benchmarks.hellaswag_ella_mind as hellaswag_ella_mind
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    Role,
)
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
@pytest.mark.parametrize("task_name", get_task_names_for_module("hellaswag_ella_mind"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "deu"

# Fictional rows following the HellaSwag format. NOT real examples from the HellaSwag dataset.
# Option order and correct letters are shuffled deterministically (seed: context+correct_ending).
_EVAL_ROW: dict[str, Any] = {
    "activity": "Kochen",
    "context": "Die Zwiebeln werden in der Pfanne angebraten",
    "correct_ending": "bis sie goldbraun sind.",
    "easy_distractors": ["mit einem Schraubenzieher.", "auf dem Dach.", "im Schwimmbad."],
    "hard_distractors": ["bis sie gefroren sind.", "bis sie roh sind.", "bis sie trocken sind."],
}

_FEWSHOT_ROW: dict[str, Any] = {
    "activity": "Sport",
    "context": "Der Spieler rennt über das Feld",
    "correct_ending": "und macht ein Tor.",
    "easy_distractors": ["und liest ein Buch.", "und kocht Suppe.", "und schläft ein."],
    "hard_distractors": ["und springt ins Wasser.", "und setzt sich hin.", "und kommt nicht an."],
}

# Expected prompts (messages, flat concat, ground truth, completions).
# --- HELLASWAG_ELLAMIND_EASY_DE ---
_EASY_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Kochen: Die Zwiebeln werden in der Pfanne angebraten"),
    ],
    concat="""\
Kochen: Die Zwiebeln werden in der Pfanne angebraten""",
    ground_truth=" bis sie goldbraun sind.",
    completions=[
        " auf dem Dach.",
        " bis sie goldbraun sind.",
        " mit einem Schraubenzieher.",
        " im Schwimmbad.",
    ],
)

_EASY_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Sport: Der Spieler rennt über das Feld"),
        Message(role=Role.ASSISTANT, content=" und macht ein Tor."),
        Message(role=Role.USER, content="Kochen: Die Zwiebeln werden in der Pfanne angebraten"),
    ],
    concat="""\
Sport: Der Spieler rennt über das Feld und macht ein Tor.

Kochen: Die Zwiebeln werden in der Pfanne angebraten""",
    ground_truth=_EASY_ZEROSHOT.ground_truth,
    completions=_EASY_ZEROSHOT.completions,
)

# --- HELLASWAG_ELLAMIND_HARD_DE ---
_HARD_ZEROSHOT = ExpectedPrompt(
    messages=_EASY_ZEROSHOT.messages,
    concat=_EASY_ZEROSHOT.concat,
    ground_truth=_EASY_ZEROSHOT.ground_truth,
    completions=[
        " bis sie roh sind.",
        " bis sie goldbraun sind.",
        " bis sie gefroren sind.",
        " bis sie trocken sind.",
    ],
)

_HARD_FEWSHOT = ExpectedPrompt(
    messages=_EASY_FEWSHOT.messages,
    concat=_EASY_FEWSHOT.concat,
    ground_truth=_EASY_ZEROSHOT.ground_truth,
    completions=_HARD_ZEROSHOT.completions,
)

# --- HELLASWAG_ELLAMIND_BPB_DE ---
# Same prompt as easy/hard; BPB scores only the gold continuation.
_easy_ground_truth = _EASY_ZEROSHOT.ground_truth
assert isinstance(_easy_ground_truth, str)  # narrow the type: easy ground_truth is always a str

_BPB_ZEROSHOT = ExpectedPrompt(
    messages=_EASY_ZEROSHOT.messages,
    concat=_EASY_ZEROSHOT.concat,
    ground_truth=_easy_ground_truth,
    completions=[_easy_ground_truth],
)

_BPB_FEWSHOT = ExpectedPrompt(
    messages=_EASY_FEWSHOT.messages,
    concat=_EASY_FEWSHOT.concat,
    ground_truth=_easy_ground_truth,
    completions=[_easy_ground_truth],
)


# --- TESTS ---
def test_hellaswag_ellamind_easy_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        hellaswag_ella_mind.HELLASWAG_ELLAMIND_EASY_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_EASY_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        hellaswag_ella_mind.HELLASWAG_ELLAMIND_EASY_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_EASY_FEWSHOT,
    )


def test_hellaswag_ellamind_hard_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        hellaswag_ella_mind.HELLASWAG_ELLAMIND_HARD_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_HARD_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        hellaswag_ella_mind.HELLASWAG_ELLAMIND_HARD_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_HARD_FEWSHOT,
    )


def test_hellaswag_ellamind_bpb_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        hellaswag_ella_mind.HELLASWAG_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        hellaswag_ella_mind.HELLASWAG_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )
