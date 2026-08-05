"""Tests for the German HLE (EllaMind) tasks.

Tests:
- formatter hash test for every HLE variant
- offline prompt assembly tests
- native multiple-choice filtering test (offline)
"""

from typing import Any

import pytest

import eval_framework.tasks.benchmarks.hle_ella_mind as hle_ella_mind
from eval_framework.tasks.base import BaseTask
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
@pytest.mark.parametrize("task_name", get_task_names_for_module("hle_ella_mind"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "deu"

# Fictional rows following the HLE format. NOT real examples from the HLE dataset.
# Option order and correct letters are shuffled deterministically (seed: question+answer).
_EVAL_ROW: dict[str, Any] = {
    "question": "Was ist die Antwort auf die Frage nach dem Leben, dem Universum und allem?",
    "correct_answer": "42",
    "incorrect_answers": ["41", "43", "44"],
    "answer_type": "multipleChoice",
}

_FEWSHOT_ROW: dict[str, Any] = {
    "question": "Was ist die Summe der Innenwinkel eines Dreiecks?",
    "correct_answer": "180",
    "incorrect_answers": ["90", "270", "360"],
    "answer_type": "multipleChoice",
}

# Expected prompts (messages, flat concat, ground truth, completions).
# --- HLE_ELLAMIND_MC_DE ---
_MC_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Was ist die Antwort auf die Frage nach dem Leben, dem Universum und allem?\nA. 44\nB. 43\nC. 41\nD. 42\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die Antwort auf die Frage nach dem Leben, dem Universum und allem?
A. 44
B. 43
C. 41
D. 42
Antwort:""",
    ground_truth=" D",
    completions=[" A", " B", " C", " D"],
)

_MC_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Was ist die Summe der Innenwinkel eines Dreiecks?\nA. 270\nB. 180\nC. 360\nD. 90\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort: B"),
        Message(
            role=Role.USER,
            content="Frage: Was ist die Antwort auf die Frage nach dem Leben, dem Universum und allem?\nA. 44\nB. 43\nC. 41\nD. 42\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die Summe der Innenwinkel eines Dreiecks?
A. 270
B. 180
C. 360
D. 90
Antwort: B

Frage: Was ist die Antwort auf die Frage nach dem Leben, dem Universum und allem?
A. 44
B. 43
C. 41
D. 42
Antwort:""",
    ground_truth=_MC_ZEROSHOT.ground_truth,
    completions=_MC_ZEROSHOT.completions,
)

# --- HLE_ELLAMIND_MC_NATIVE_DE ---
# Same prompt render as MC.
_MC_NATIVE_ZEROSHOT = ExpectedPrompt(
    messages=_MC_ZEROSHOT.messages,
    concat=_MC_ZEROSHOT.concat,
    ground_truth=_MC_ZEROSHOT.ground_truth,
    completions=_MC_ZEROSHOT.completions,
)

_MC_NATIVE_FEWSHOT = ExpectedPrompt(
    messages=_MC_FEWSHOT.messages,
    concat=_MC_FEWSHOT.concat,
    ground_truth=_MC_ZEROSHOT.ground_truth,
    completions=_MC_ZEROSHOT.completions,
)

# --- HLE_ELLAMIND_CLOZE_DE ---
_CLOZE_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Was ist die Antwort auf die Frage nach dem Leben, dem Universum und allem?\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die Antwort auf die Frage nach dem Leben, dem Universum und allem?
Antwort:""",
    ground_truth=" 42",
    completions=[" 44", " 43", " 41", " 42"],
)

_CLOZE_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Frage: Was ist die Summe der Innenwinkel eines Dreiecks?\n"),
        Message(role=Role.ASSISTANT, content="Antwort: 180"),
        Message(
            role=Role.USER,
            content="Frage: Was ist die Antwort auf die Frage nach dem Leben, dem Universum und allem?\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die Summe der Innenwinkel eines Dreiecks?
Antwort: 180

Frage: Was ist die Antwort auf die Frage nach dem Leben, dem Universum und allem?
Antwort:""",
    ground_truth=_CLOZE_ZEROSHOT.ground_truth,
    completions=_CLOZE_ZEROSHOT.completions,
)

# --- HLE_ELLAMIND_CLOZE_NATIVE_DE ---
# Same prompt render as Cloze.
_CLOZE_NATIVE_ZEROSHOT = ExpectedPrompt(
    messages=_CLOZE_ZEROSHOT.messages,
    concat=_CLOZE_ZEROSHOT.concat,
    ground_truth=_CLOZE_ZEROSHOT.ground_truth,
    completions=_CLOZE_ZEROSHOT.completions,
)

_CLOZE_NATIVE_FEWSHOT = ExpectedPrompt(
    messages=_CLOZE_FEWSHOT.messages,
    concat=_CLOZE_FEWSHOT.concat,
    ground_truth=_CLOZE_ZEROSHOT.ground_truth,
    completions=_CLOZE_ZEROSHOT.completions,
)

# --- HLE_ELLAMIND_BPB_DE ---
# Same prompt as cloze; BPB scores only the gold continuation.
_cloze_ground_truth = _CLOZE_ZEROSHOT.ground_truth
assert isinstance(_cloze_ground_truth, str)  # narrow the type: cloze ground_truth is always a str

_BPB_ZEROSHOT = ExpectedPrompt(
    messages=_CLOZE_ZEROSHOT.messages,
    concat=_CLOZE_ZEROSHOT.concat,
    ground_truth=_cloze_ground_truth,
    completions=[_cloze_ground_truth],
)

_BPB_FEWSHOT = ExpectedPrompt(
    messages=_CLOZE_FEWSHOT.messages,
    concat=_CLOZE_FEWSHOT.concat,
    ground_truth=_cloze_ground_truth,
    completions=[_cloze_ground_truth],
)


# --- TESTS ---
def test_hle_ellamind_mc_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        hle_ella_mind.HLE_ELLAMIND_MC_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        hle_ella_mind.HLE_ELLAMIND_MC_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_FEWSHOT,
    )


def test_hle_ellamind_mc_native_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        hle_ella_mind.HLE_ELLAMIND_MC_NATIVE_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_NATIVE_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        hle_ella_mind.HLE_ELLAMIND_MC_NATIVE_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_NATIVE_FEWSHOT,
    )


def test_hle_ellamind_cloze_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        hle_ella_mind.HLE_ELLAMIND_CLOZE_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        hle_ella_mind.HLE_ELLAMIND_CLOZE_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_FEWSHOT,
    )


def test_hle_ellamind_cloze_native_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        hle_ella_mind.HLE_ELLAMIND_CLOZE_NATIVE_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_NATIVE_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        hle_ella_mind.HLE_ELLAMIND_CLOZE_NATIVE_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_NATIVE_FEWSHOT,
    )


def test_hle_ellamind_bpb_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        hle_ella_mind.HLE_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        hle_ella_mind.HLE_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )


# ---------------------------------------------------------------------------
# Native multiple-choice filtering test (offline)
# ---------------------------------------------------------------------------


def test_hle_native_variant_filters_to_multiple_choice_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native HLE variants should keep only `answer_type == "multipleChoice"`."""

    def fake_base_load_dataset(self: BaseTask, subject: str) -> None:
        _ = subject
        self.dataset = {
            self.SAMPLE_SPLIT: [
                {"answer_type": "multipleChoice", "question": "Q1"},
                {"answer_type": "shortAnswer", "question": "Q2"},
                {"answer_type": "multipleChoice", "question": "Q3"},
            ]
        }

    monkeypatch.setattr(BaseTask, "_load_dataset", fake_base_load_dataset)
    task = hle_ella_mind.HLE_ELLAMIND_MC_NATIVE_DE(num_fewshot=0)
    task._load_dataset("deu")

    assert len(task.dataset[task.SAMPLE_SPLIT]) == 2
    assert all(item["answer_type"] == "multipleChoice" for item in task.dataset[task.SAMPLE_SPLIT])
