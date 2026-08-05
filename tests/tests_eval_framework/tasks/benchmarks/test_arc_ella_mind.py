"""Tests for the German ARC (EllaMind) tasks.

Tests:
- formatter hash test for every ARC variant
- offline prompt assembly tests
- dataset loading and filtering tests (offline)
"""

from typing import Any

import pytest

import eval_framework.tasks.benchmarks.arc_ella_mind as arc_ella_mind
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
@pytest.mark.parametrize("task_name", get_task_names_for_module("arc_ella_mind"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "ARC-Easy"

# Fictional rows following the ARC format. NOT real examples from the ARC dataset.
_EVAL_ROW: dict[str, Any] = {
    "question": "Was ist die Hauptstadt von Frankreich?",
    "choices": ["Berlin", "Paris", "London", "Madrid"],
    "answer_key": "B",
    "arc_config": _SUBJECT,
}

_FEWSHOT_ROW: dict[str, Any] = {
    "question": "Was ist 1+1?",
    "choices": ["0", "1", "2", "3"],
    "answer_key": "C",
    "arc_config": _SUBJECT,
}

# Expected prompts (messages, flat concat, ground truth, completions).
# --- ARC_ELLAMIND_MC_DE ---
_MC_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Was ist die Hauptstadt von Frankreich?\nA. Berlin\nB. Paris\nC. London\nD. Madrid\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die Hauptstadt von Frankreich?
A. Berlin
B. Paris
C. London
D. Madrid
Antwort:""",
    ground_truth=" B",
    completions=[" A", " B", " C", " D"],
)

_MC_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Frage: Was ist 1+1?\nA. 0\nB. 1\nC. 2\nD. 3\n"),
        Message(role=Role.ASSISTANT, content="Antwort: C"),
        Message(
            role=Role.USER,
            content="Frage: Was ist die Hauptstadt von Frankreich?\nA. Berlin\nB. Paris\nC. London\nD. Madrid\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist 1+1?
A. 0
B. 1
C. 2
D. 3
Antwort: C

Frage: Was ist die Hauptstadt von Frankreich?
A. Berlin
B. Paris
C. London
D. Madrid
Antwort:""",
    ground_truth=_MC_ZEROSHOT.ground_truth,
    completions=_MC_ZEROSHOT.completions,
)

# --- ARC_ELLAMIND_CLOZE_DE ---
_CLOZE_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Frage: Was ist die Hauptstadt von Frankreich?\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die Hauptstadt von Frankreich?
Antwort:""",
    ground_truth=" Paris",
    completions=[" Berlin", " Paris", " London", " Madrid"],
)

_CLOZE_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Frage: Was ist 1+1?\n"),
        Message(role=Role.ASSISTANT, content="Antwort: 2"),
        Message(role=Role.USER, content="Frage: Was ist die Hauptstadt von Frankreich?\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist 1+1?
Antwort: 2

Frage: Was ist die Hauptstadt von Frankreich?
Antwort:""",
    ground_truth=_CLOZE_ZEROSHOT.ground_truth,
    completions=_CLOZE_ZEROSHOT.completions,
)

# --- ARC_ELLAMIND_BPB_DE ---
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
def test_arc_ellamind_mc_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        arc_ella_mind.ARC_ELLAMIND_MC_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        arc_ella_mind.ARC_ELLAMIND_MC_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_FEWSHOT,
    )


def test_arc_ellamind_cloze_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        arc_ella_mind.ARC_ELLAMIND_CLOZE_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        arc_ella_mind.ARC_ELLAMIND_CLOZE_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_FEWSHOT,
    )


def test_arc_ellamind_bpb_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        arc_ella_mind.ARC_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        arc_ella_mind.ARC_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )


# ---------------------------------------------------------------------------
# Dataset loading and filtering tests (offline)
# ---------------------------------------------------------------------------


def test_arc_ellamind_uses_german_config_and_filters_arc_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test checks that the dataset loading works as expected.

    Specifically, it checks that:
    1) The dataset loads only the German subset (``deu``).
    2) The dataset filters rows to the requested subset (``ARC-Easy`` or ``ARC-Challenge``).
    """
    called_subjects: list[str] = []

    def fake_base_load_dataset(self: BaseTask, subject: str) -> None:
        called_subjects.append(subject)
        self.dataset = {
            self.SAMPLE_SPLIT: [
                {"arc_config": "ARC-Easy", "question": "Q1"},
                {"arc_config": "ARC-Challenge", "question": "Q2"},
                {"arc_config": "ARC-Easy", "question": "Q3"},
            ]
        }

    monkeypatch.setattr(BaseTask, "_load_dataset", fake_base_load_dataset)
    task = arc_ella_mind.ARC_ELLAMIND_CLOZE_DE(num_fewshot=0)

    task._load_dataset("ARC-Easy")
    assert called_subjects[-1] == "deu"
    assert [item["question"] for item in task.dataset[task.SAMPLE_SPLIT]] == [
        "Q1",
        "Q3",
    ]
    assert all(item["arc_config"] == "ARC-Easy" for item in task.dataset[task.SAMPLE_SPLIT])

    task._load_dataset("ARC-Challenge")
    assert called_subjects[-1] == "deu"
    assert [item["question"] for item in task.dataset[task.SAMPLE_SPLIT]] == ["Q2"]
    assert all(item["arc_config"] == "ARC-Challenge" for item in task.dataset[task.SAMPLE_SPLIT])
