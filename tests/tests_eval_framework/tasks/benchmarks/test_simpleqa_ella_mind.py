"""Tests for the German SimpleQA (EllaMind) tasks.

Tests:
- formatter hash test for every SimpleQA variant
- offline prompt assembly tests
"""

from typing import Any

import pytest

import eval_framework.tasks.benchmarks.simpleqa_ella_mind as simpleqa_ella_mind
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
@pytest.mark.parametrize("task_name", get_task_names_for_module("simpleqa_ella_mind"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "deu"

# Fictional rows following the SimpleQA format. NOT real examples from the SimpleQA dataset.
# Option order and correct letters are shuffled deterministically (seed: question+answer).
_EVAL_ROW: dict[str, Any] = {
    "question": "Welches Jahr haben?",
    "answer": "2026",
    "easy_distractors": ["1954", "1974", "1990"],
    "hard_distractors": ["2024", "2025", "2027"],
}

_FEWSHOT_ROW: dict[str, Any] = {
    "question": "Was ist die Hauptstadt von Frankreich?",
    "answer": "Paris",
    "easy_distractors": ["London", "Berlin", "Madrid"],
    "hard_distractors": ["Lyon", "Bordeaux", "Marseille"],
}

# Expected prompts (messages, flat concat, ground truth, completions).
# --- SIMPLEQA_ELLAMIND_MC_EASY_DE ---
_MC_EASY_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Welches Jahr haben?\nA. 1990\nB. 1954\nC. 1974\nD. 2026\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Welches Jahr haben?
A. 1990
B. 1954
C. 1974
D. 2026
Antwort:""",
    ground_truth=" D",
    completions=[" A", " B", " C", " D"],
)

_MC_EASY_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Was ist die Hauptstadt von Frankreich?\nA. Paris\nB. Berlin\nC. London\nD. Madrid\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort: A"),
        Message(
            role=Role.USER,
            content="Frage: Welches Jahr haben?\nA. 1990\nB. 1954\nC. 1974\nD. 2026\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die Hauptstadt von Frankreich?
A. Paris
B. Berlin
C. London
D. Madrid
Antwort: A

Frage: Welches Jahr haben?
A. 1990
B. 1954
C. 1974
D. 2026
Antwort:""",
    ground_truth=_MC_EASY_ZEROSHOT.ground_truth,
    completions=_MC_EASY_ZEROSHOT.completions,
)

# --- SIMPLEQA_ELLAMIND_MC_HARD_DE ---
_MC_HARD_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Welches Jahr haben?\nA. 2027\nB. 2024\nC. 2025\nD. 2026\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Welches Jahr haben?
A. 2027
B. 2024
C. 2025
D. 2026
Antwort:""",
    ground_truth=_MC_EASY_ZEROSHOT.ground_truth,
    completions=_MC_EASY_ZEROSHOT.completions,
)

_MC_HARD_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Was ist die Hauptstadt von Frankreich?\nA. Paris\nB. Bordeaux\nC. Lyon\nD. Marseille\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort: A"),
        Message(
            role=Role.USER,
            content="Frage: Welches Jahr haben?\nA. 2027\nB. 2024\nC. 2025\nD. 2026\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die Hauptstadt von Frankreich?
A. Paris
B. Bordeaux
C. Lyon
D. Marseille
Antwort: A

Frage: Welches Jahr haben?
A. 2027
B. 2024
C. 2025
D. 2026
Antwort:""",
    ground_truth=_MC_HARD_ZEROSHOT.ground_truth,
    completions=_MC_HARD_ZEROSHOT.completions,
)

# --- SIMPLEQA_ELLAMIND_CLOZE_EASY_DE ---
# Cloze prompts show no options, so the easy/hard variants share the same prompt; only the
# scored completions differ.
_CLOZE_EASY_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Frage: Welches Jahr haben?\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Welches Jahr haben?
Antwort:""",
    ground_truth=" 2026",
    completions=[" 1990", " 1954", " 1974", " 2026"],
)

_CLOZE_EASY_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Frage: Was ist die Hauptstadt von Frankreich?\n"),
        Message(role=Role.ASSISTANT, content="Antwort: Paris"),
        Message(role=Role.USER, content="Frage: Welches Jahr haben?\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Was ist die Hauptstadt von Frankreich?
Antwort: Paris

Frage: Welches Jahr haben?
Antwort:""",
    ground_truth=_CLOZE_EASY_ZEROSHOT.ground_truth,
    completions=_CLOZE_EASY_ZEROSHOT.completions,
)

# --- SIMPLEQA_ELLAMIND_CLOZE_HARD_DE ---
# Same prompt as cloze-easy; only the (hard) distractor completions differ.
_CLOZE_HARD_ZEROSHOT = ExpectedPrompt(
    messages=_CLOZE_EASY_ZEROSHOT.messages,
    concat=_CLOZE_EASY_ZEROSHOT.concat,
    ground_truth=_CLOZE_EASY_ZEROSHOT.ground_truth,
    completions=[" 2027", " 2024", " 2025", " 2026"],
)

_CLOZE_HARD_FEWSHOT = ExpectedPrompt(
    messages=_CLOZE_EASY_FEWSHOT.messages,
    concat=_CLOZE_EASY_FEWSHOT.concat,
    ground_truth=_CLOZE_HARD_ZEROSHOT.ground_truth,
    completions=_CLOZE_HARD_ZEROSHOT.completions,
)

# --- SIMPLEQA_ELLAMIND_BPB_DE ---
# Same prompt as cloze-easy; BPB scores only the gold continuation.
_cloze_ground_truth = _CLOZE_EASY_ZEROSHOT.ground_truth
assert isinstance(_cloze_ground_truth, str)  # narrow the type: cloze ground_truth is always a str

_BPB_ZEROSHOT = ExpectedPrompt(
    messages=_CLOZE_EASY_ZEROSHOT.messages,
    concat=_CLOZE_EASY_ZEROSHOT.concat,
    ground_truth=_cloze_ground_truth,
    completions=[_cloze_ground_truth],
)

_BPB_FEWSHOT = ExpectedPrompt(
    messages=_CLOZE_EASY_FEWSHOT.messages,
    concat=_CLOZE_EASY_FEWSHOT.concat,
    ground_truth=_cloze_ground_truth,
    completions=[_cloze_ground_truth],
)


# --- TESTS ---
def test_simpleqa_ellamind_mc_easy_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        simpleqa_ella_mind.SIMPLEQA_ELLAMIND_MC_EASY_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_EASY_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        simpleqa_ella_mind.SIMPLEQA_ELLAMIND_MC_EASY_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_EASY_FEWSHOT,
    )


def test_simpleqa_ellamind_mc_hard_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        simpleqa_ella_mind.SIMPLEQA_ELLAMIND_MC_HARD_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_HARD_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        simpleqa_ella_mind.SIMPLEQA_ELLAMIND_MC_HARD_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_HARD_FEWSHOT,
    )


def test_simpleqa_ellamind_cloze_easy_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        simpleqa_ella_mind.SIMPLEQA_ELLAMIND_CLOZE_EASY_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_EASY_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        simpleqa_ella_mind.SIMPLEQA_ELLAMIND_CLOZE_EASY_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_EASY_FEWSHOT,
    )


def test_simpleqa_ellamind_cloze_hard_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        simpleqa_ella_mind.SIMPLEQA_ELLAMIND_CLOZE_HARD_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_HARD_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        simpleqa_ella_mind.SIMPLEQA_ELLAMIND_CLOZE_HARD_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_HARD_FEWSHOT,
    )


def test_simpleqa_ellamind_bpb_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        simpleqa_ella_mind.SIMPLEQA_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        simpleqa_ella_mind.SIMPLEQA_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )
