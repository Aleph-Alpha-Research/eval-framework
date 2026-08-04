"""Tests for the German GPQA (EllaMind) tasks.

Tests:
- formatter hash test for every GPQA variant
- offline prompt assembly tests
- diamond-subset filtering test (offline)
"""

from typing import Any

import pytest

import eval_framework.tasks.benchmarks.gpqa_ellamind as gpqa_ellamind
from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_gpqa_ellamind_tasks
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
    run_formatter_hash_test,
)

# Registry for this test suite only holding gpqa_ellamind tasks
_gpqa_ellamind_registry = Registry()
register_gpqa_ellamind_tasks(registry=_gpqa_ellamind_registry)

# ---------------------------------------------------------------------------
# Formatter hash tests (Hugging Face)
# ---------------------------------------------------------------------------


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _gpqa_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_gpqa_ellamind_registry)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "deu"

# Fictional rows following the GPQA format. NOT real examples from the GPQA dataset.
# Option order and correct letters are shuffled deterministically (seed: question+answer).
_EVAL_ROW: dict[str, Any] = {
    "question": "Was ist die SI-Einheit des elektrischen Widerstands?",
    "correct_answer": "Ohm",
    "incorrect_answers": ["Volt", "Ampere", "Watt"],
    "is_diamond": True,
}

_FEWSHOT_ROW: dict[str, Any] = {
    "question": "Was ist die SI-Einheit der Temperatur?",
    "correct_answer": "Kelvin",
    "incorrect_answers": ["Celsius", "Fahrenheit", "Joule"],
    "is_diamond": True,
}

# Expected prompts (messages, flat concat, ground truth, completions).
# --- GPQA_ELLAMIND_MC_DE ---
_MC_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Was ist die SI-Einheit des elektrischen Widerstands?\nA. Ohm\nB. Volt\nC. Watt\nD. Ampere\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die SI-Einheit des elektrischen Widerstands?
A. Ohm
B. Volt
C. Watt
D. Ampere
Antwort:""",
    ground_truth=" A",
    completions=[" A", " B", " C", " D"],
)

_MC_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Was ist die SI-Einheit der Temperatur?\nA. Celsius\nB. Fahrenheit\nC. Joule\nD. Kelvin\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort: D"),
        Message(
            role=Role.USER,
            content="Frage: Was ist die SI-Einheit des elektrischen Widerstands?\nA. Ohm\nB. Volt\nC. Watt\nD. Ampere\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die SI-Einheit der Temperatur?
A. Celsius
B. Fahrenheit
C. Joule
D. Kelvin
Antwort: D

Frage: Was ist die SI-Einheit des elektrischen Widerstands?
A. Ohm
B. Volt
C. Watt
D. Ampere
Antwort:""",
    ground_truth=_MC_ZEROSHOT.ground_truth,
    completions=_MC_ZEROSHOT.completions,
)

# --- GPQA_ELLAMIND_CLOZE_DE ---
_CLOZE_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Frage: Was ist die SI-Einheit des elektrischen Widerstands?\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die SI-Einheit des elektrischen Widerstands?
Antwort:""",
    ground_truth=" Ohm",
    completions=[" Ohm", " Volt", " Watt", " Ampere"],
)

_CLOZE_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Frage: Was ist die SI-Einheit der Temperatur?\n"),
        Message(role=Role.ASSISTANT, content="Antwort: Kelvin"),
        Message(role=Role.USER, content="Frage: Was ist die SI-Einheit des elektrischen Widerstands?\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die SI-Einheit der Temperatur?
Antwort: Kelvin

Frage: Was ist die SI-Einheit des elektrischen Widerstands?
Antwort:""",
    ground_truth=_CLOZE_ZEROSHOT.ground_truth,
    completions=_CLOZE_ZEROSHOT.completions,
)

# --- GPQA_ELLAMIND_BPB_DE ---
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
def test_gpqa_ellamind_mc_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        gpqa_ellamind.GPQA_ELLAMIND_MC_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        gpqa_ellamind.GPQA_ELLAMIND_MC_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_FEWSHOT,
    )


def test_gpqa_ellamind_diamond_mc_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        gpqa_ellamind.GPQA_ELLAMIND_DIAMOND_MC_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        gpqa_ellamind.GPQA_ELLAMIND_DIAMOND_MC_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_FEWSHOT,
    )


def test_gpqa_ellamind_cloze_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        gpqa_ellamind.GPQA_ELLAMIND_CLOZE_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        gpqa_ellamind.GPQA_ELLAMIND_CLOZE_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_FEWSHOT,
    )


def test_gpqa_ellamind_diamond_cloze_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        gpqa_ellamind.GPQA_ELLAMIND_DIAMOND_CLOZE_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        gpqa_ellamind.GPQA_ELLAMIND_DIAMOND_CLOZE_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_FEWSHOT,
    )


def test_gpqa_ellamind_bpb_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        gpqa_ellamind.GPQA_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        gpqa_ellamind.GPQA_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )


def test_gpqa_ellamind_diamond_bpb_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        gpqa_ellamind.GPQA_ELLAMIND_DIAMOND_BPB_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        gpqa_ellamind.GPQA_ELLAMIND_DIAMOND_BPB_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )


# ---------------------------------------------------------------------------
# Diamond-subset filtering test (offline)
# ---------------------------------------------------------------------------


def test_gpqa_diamond_variant_filters_to_diamond_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diamond variant should keep only rows with `is_diamond=True`."""

    def fake_base_load_dataset(self: BaseTask, subject: str) -> None:
        _ = subject
        self.dataset = {
            self.SAMPLE_SPLIT: [
                {"is_diamond": True, "question": "Q1"},
                {"is_diamond": False, "question": "Q2"},
                {"is_diamond": True, "question": "Q3"},
            ]
        }

    monkeypatch.setattr(BaseTask, "_load_dataset", fake_base_load_dataset)
    task = gpqa_ellamind.GPQA_ELLAMIND_DIAMOND_MC_DE(num_fewshot=0)
    task._load_dataset("deu")

    assert len(task.dataset[task.SAMPLE_SPLIT]) == 2
    assert all(item["is_diamond"] for item in task.dataset[task.SAMPLE_SPLIT])
