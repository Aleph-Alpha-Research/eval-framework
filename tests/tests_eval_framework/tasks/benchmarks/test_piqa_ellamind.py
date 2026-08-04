"""Tests for the German PIQA (EllaMind) tasks.

Tests:
- formatter hash test for every PIQA variant
- offline prompt assembly tests
"""

from typing import Any

import pytest

import eval_framework.tasks.benchmarks.piqa_ellamind as piqa_ellamind
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_piqa_ellamind_tasks
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

# Registry for this test suite only holding piqa_ellamind tasks
_piqa_ellamind_registry = Registry()
register_piqa_ellamind_tasks(registry=_piqa_ellamind_registry)

# ---------------------------------------------------------------------------
# Formatter hash tests (Hugging Face)
# ---------------------------------------------------------------------------


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _piqa_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_piqa_ellamind_registry)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "deu"

# Fictional rows following the PIQA format. NOT real examples from the PIQA dataset.
# Option order and correct letters are shuffled deterministically (seed: question+answer).
_EVAL_ROW: dict[str, Any] = {
    "goal": "Martin möchte einen Nagel in die Wand schlagen.",
    "correct_solution": "Er verwendet einen Hammer.",
    "easy_distractor": "Er verwendet eine Schere.",
    "hard_distractor": "Er verwendet eine Zange.",
}

_FEWSHOT_ROW: dict[str, Any] = {
    "goal": "Prabhu will Wasser aufkochen.",
    "correct_solution": "Er stellt den Topf auf den Herd.",
    "easy_distractor": "Er schreit den Topf an, bis er warm wird.",
    "hard_distractor": "Er stellt den Topf über Nacht in den Kühlschrank.",
}

# Expected prompts (messages, flat concat, ground truth, completions).
# --- PIQA_ELLAMIND_MC_EASY_DE ---
_MC_EASY_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Ziel: Martin möchte einen Nagel in die Wand schlagen.\nA. Er verwendet einen Hammer.\nB. Er verwendet eine Schere.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Ziel: Martin möchte einen Nagel in die Wand schlagen.
A. Er verwendet einen Hammer.
B. Er verwendet eine Schere.
Antwort:""",
    ground_truth=" A",
    completions=[" A", " B"],
)

_MC_EASY_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Ziel: Prabhu will Wasser aufkochen.\nA. Er stellt den Topf auf den Herd.\nB. Er schreit den Topf an, bis er warm wird.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort: A"),
        Message(
            role=Role.USER,
            content="Ziel: Martin möchte einen Nagel in die Wand schlagen.\nA. Er verwendet einen Hammer.\nB. Er verwendet eine Schere.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Ziel: Prabhu will Wasser aufkochen.
A. Er stellt den Topf auf den Herd.
B. Er schreit den Topf an, bis er warm wird.
Antwort: A

Ziel: Martin möchte einen Nagel in die Wand schlagen.
A. Er verwendet einen Hammer.
B. Er verwendet eine Schere.
Antwort:""",
    ground_truth=_MC_EASY_ZEROSHOT.ground_truth,
    completions=_MC_EASY_ZEROSHOT.completions,
)

# --- PIQA_ELLAMIND_MC_HARD_DE ---
_MC_HARD_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Ziel: Martin möchte einen Nagel in die Wand schlagen.\nA. Er verwendet einen Hammer.\nB. Er verwendet eine Zange.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Ziel: Martin möchte einen Nagel in die Wand schlagen.
A. Er verwendet einen Hammer.
B. Er verwendet eine Zange.
Antwort:""",
    ground_truth=_MC_EASY_ZEROSHOT.ground_truth,
    completions=_MC_EASY_ZEROSHOT.completions,
)

_MC_HARD_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Ziel: Prabhu will Wasser aufkochen.\nA. Er stellt den Topf auf den Herd.\nB. Er stellt den Topf über Nacht in den Kühlschrank.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort: A"),
        Message(
            role=Role.USER,
            content="Ziel: Martin möchte einen Nagel in die Wand schlagen.\nA. Er verwendet einen Hammer.\nB. Er verwendet eine Zange.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Ziel: Prabhu will Wasser aufkochen.
A. Er stellt den Topf auf den Herd.
B. Er stellt den Topf über Nacht in den Kühlschrank.
Antwort: A

Ziel: Martin möchte einen Nagel in die Wand schlagen.
A. Er verwendet einen Hammer.
B. Er verwendet eine Zange.
Antwort:""",
    ground_truth=_MC_HARD_ZEROSHOT.ground_truth,
    completions=_MC_HARD_ZEROSHOT.completions,
)

# --- PIQA_ELLAMIND_CLOZE_EASY_DE ---
# Cloze prompts show no options, so the easy/hard variants share the same prompt; only the
# scored completions differ.
_CLOZE_EASY_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Ziel: Martin möchte einen Nagel in die Wand schlagen.\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Ziel: Martin möchte einen Nagel in die Wand schlagen.
Antwort:""",
    ground_truth=" Er verwendet einen Hammer.",
    completions=[
        " Er verwendet einen Hammer.",
        " Er verwendet eine Schere.",
    ],
)

_CLOZE_EASY_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Ziel: Prabhu will Wasser aufkochen.\n"),
        Message(role=Role.ASSISTANT, content="Antwort: Er stellt den Topf auf den Herd."),
        Message(role=Role.USER, content="Ziel: Martin möchte einen Nagel in die Wand schlagen.\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Ziel: Prabhu will Wasser aufkochen.
Antwort: Er stellt den Topf auf den Herd.

Ziel: Martin möchte einen Nagel in die Wand schlagen.
Antwort:""",
    ground_truth=_CLOZE_EASY_ZEROSHOT.ground_truth,
    completions=_CLOZE_EASY_ZEROSHOT.completions,
)

# --- PIQA_ELLAMIND_CLOZE_HARD_DE ---
# Same prompt as cloze-easy; only the (hard) distractor completion differs.
_CLOZE_HARD_ZEROSHOT = ExpectedPrompt(
    messages=_CLOZE_EASY_ZEROSHOT.messages,
    concat=_CLOZE_EASY_ZEROSHOT.concat,
    ground_truth=_CLOZE_EASY_ZEROSHOT.ground_truth,
    completions=[
        " Er verwendet einen Hammer.",
        " Er verwendet eine Zange.",
    ],
)

_CLOZE_HARD_FEWSHOT = ExpectedPrompt(
    messages=_CLOZE_EASY_FEWSHOT.messages,
    concat=_CLOZE_EASY_FEWSHOT.concat,
    ground_truth=_CLOZE_EASY_ZEROSHOT.ground_truth,
    completions=_CLOZE_HARD_ZEROSHOT.completions,
)

# --- PIQA_ELLAMIND_BPB_DE ---
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
def test_piqa_ellamind_mc_easy_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        piqa_ellamind.PIQA_ELLAMIND_MC_EASY_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_EASY_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        piqa_ellamind.PIQA_ELLAMIND_MC_EASY_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_EASY_FEWSHOT,
    )


def test_piqa_ellamind_mc_hard_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        piqa_ellamind.PIQA_ELLAMIND_MC_HARD_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_HARD_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        piqa_ellamind.PIQA_ELLAMIND_MC_HARD_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_HARD_FEWSHOT,
    )


def test_piqa_ellamind_cloze_easy_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        piqa_ellamind.PIQA_ELLAMIND_CLOZE_EASY_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_EASY_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        piqa_ellamind.PIQA_ELLAMIND_CLOZE_EASY_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_EASY_FEWSHOT,
    )


def test_piqa_ellamind_cloze_hard_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        piqa_ellamind.PIQA_ELLAMIND_CLOZE_HARD_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_HARD_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        piqa_ellamind.PIQA_ELLAMIND_CLOZE_HARD_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_HARD_FEWSHOT,
    )


def test_piqa_ellamind_bpb_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        piqa_ellamind.PIQA_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        piqa_ellamind.PIQA_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )
