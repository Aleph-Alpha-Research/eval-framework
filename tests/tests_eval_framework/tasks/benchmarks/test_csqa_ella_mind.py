"""Tests for the German CSQA (EllaMind) tasks.

Tests:
- formatter hash test for every CSQA variant
- offline prompt assembly tests
- distractor shuffling tests (offline)
"""

from typing import Any

import pytest

import eval_framework.tasks.benchmarks.csqa_ella_mind as csqa_ella_mind
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
@pytest.mark.parametrize("task_name", get_task_names_for_module("csqa_ella_mind"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "deu"

# Fictional rows following the CSQA format. NOT real examples from the CSQA dataset.
# Option order and correct letters are shuffled deterministically (seed: question+answer).
_EVAL_ROW: dict[str, Any] = {
    "question": "Wo bewahrt man frische Milch am besten auf?",
    "correct_answer": "Im Kühlschrank",
    "easy_distractors": ["Auf dem Mond", "In einem Schuh", "Im Vulkan"],
    "hard_distractors": ["In der Speisekammer", "Auf der Fensterbank", "Im Keller"],
}

_FEWSHOT_ROW: dict[str, Any] = {
    "question": "Womit schreibt man normalerweise auf Papier?",
    "correct_answer": "Mit einem Stift",
    "easy_distractors": ["Mit einer Banane", "Mit einer Wolke", "Mit einem Stein"],
    "hard_distractors": ["Mit einem Pinsel", "Mit Kreide", "Mit einer Tastatur"],
}

# Expected prompts (messages, flat concat, ground truth, completions).
# --- CSQA_ELLAMIND_MC_EASY_DE ---
_MC_EASY_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Wo bewahrt man frische Milch am besten auf?\nA. Im Kühlschrank\nB. Auf dem Mond\nC. In einem Schuh\nD. Im Vulkan\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Wo bewahrt man frische Milch am besten auf?
A. Im Kühlschrank
B. Auf dem Mond
C. In einem Schuh
D. Im Vulkan
Antwort:""",
    ground_truth=" A",
    completions=[" A", " B", " C", " D"],
)

_MC_EASY_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Womit schreibt man normalerweise auf Papier?\nA. Mit einer Wolke\nB. Mit einem Stift\nC. Mit einem Stein\nD. Mit einer Banane\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort: B"),
        Message(
            role=Role.USER,
            content="Frage: Wo bewahrt man frische Milch am besten auf?\nA. Im Kühlschrank\nB. Auf dem Mond\nC. In einem Schuh\nD. Im Vulkan\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Womit schreibt man normalerweise auf Papier?
A. Mit einer Wolke
B. Mit einem Stift
C. Mit einem Stein
D. Mit einer Banane
Antwort: B

Frage: Wo bewahrt man frische Milch am besten auf?
A. Im Kühlschrank
B. Auf dem Mond
C. In einem Schuh
D. Im Vulkan
Antwort:""",
    ground_truth=_MC_EASY_ZEROSHOT.ground_truth,
    completions=_MC_EASY_ZEROSHOT.completions,
)

# --- CSQA_ELLAMIND_MC_HARD_DE ---
_MC_HARD_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Wo bewahrt man frische Milch am besten auf?\nA. Im Kühlschrank\nB. In der Speisekammer\nC. Auf der Fensterbank\nD. Im Keller\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Wo bewahrt man frische Milch am besten auf?
A. Im Kühlschrank
B. In der Speisekammer
C. Auf der Fensterbank
D. Im Keller
Antwort:""",
    ground_truth=_MC_EASY_ZEROSHOT.ground_truth,
    completions=_MC_EASY_ZEROSHOT.completions,
)

_MC_HARD_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Womit schreibt man normalerweise auf Papier?\nA. Mit Kreide\nB. Mit einem Stift\nC. Mit einer Tastatur\nD. Mit einem Pinsel\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort: B"),
        Message(
            role=Role.USER,
            content="Frage: Wo bewahrt man frische Milch am besten auf?\nA. Im Kühlschrank\nB. In der Speisekammer\nC. Auf der Fensterbank\nD. Im Keller\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Womit schreibt man normalerweise auf Papier?
A. Mit Kreide
B. Mit einem Stift
C. Mit einer Tastatur
D. Mit einem Pinsel
Antwort: B

Frage: Wo bewahrt man frische Milch am besten auf?
A. Im Kühlschrank
B. In der Speisekammer
C. Auf der Fensterbank
D. Im Keller
Antwort:""",
    ground_truth=_MC_EASY_ZEROSHOT.ground_truth,
    completions=_MC_EASY_ZEROSHOT.completions,
)

# --- CSQA_ELLAMIND_CLOZE_EASY_DE ---
# Cloze prompts show no options, so the easy/hard variants share the same prompt; only the
# scored completions differ.
_CLOZE_EASY_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Frage: Wo bewahrt man frische Milch am besten auf?\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Wo bewahrt man frische Milch am besten auf?
Antwort:""",
    ground_truth=" Im Kühlschrank",
    completions=[
        " Im Kühlschrank",
        " Auf dem Mond",
        " In einem Schuh",
        " Im Vulkan",
    ],
)

_CLOZE_EASY_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Frage: Womit schreibt man normalerweise auf Papier?\n"),
        Message(role=Role.ASSISTANT, content="Antwort: Mit einem Stift"),
        Message(role=Role.USER, content="Frage: Wo bewahrt man frische Milch am besten auf?\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Womit schreibt man normalerweise auf Papier?
Antwort: Mit einem Stift

Frage: Wo bewahrt man frische Milch am besten auf?
Antwort:""",
    ground_truth=_CLOZE_EASY_ZEROSHOT.ground_truth,
    completions=_CLOZE_EASY_ZEROSHOT.completions,
)

# --- CSQA_ELLAMIND_CLOZE_HARD_DE ---
# Same prompt as cloze-easy; only the (hard) distractor completions differ.
_CLOZE_HARD_ZEROSHOT = ExpectedPrompt(
    messages=_CLOZE_EASY_ZEROSHOT.messages,
    concat=_CLOZE_EASY_ZEROSHOT.concat,
    ground_truth=_CLOZE_EASY_ZEROSHOT.ground_truth,
    completions=[
        " Im Kühlschrank",
        " In der Speisekammer",
        " Auf der Fensterbank",
        " Im Keller",
    ],
)

_CLOZE_HARD_FEWSHOT = ExpectedPrompt(
    messages=_CLOZE_EASY_FEWSHOT.messages,
    concat=_CLOZE_EASY_FEWSHOT.concat,
    ground_truth=_CLOZE_EASY_ZEROSHOT.ground_truth,
    completions=_CLOZE_HARD_ZEROSHOT.completions,
)

# --- CSQA_ELLAMIND_BPB_DE ---
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
def test_csqa_ellamind_mc_easy_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        csqa_ella_mind.CSQA_ELLAMIND_MC_EASY_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_EASY_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        csqa_ella_mind.CSQA_ELLAMIND_MC_EASY_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_EASY_FEWSHOT,
    )


def test_csqa_ellamind_mc_hard_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        csqa_ella_mind.CSQA_ELLAMIND_MC_HARD_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_HARD_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        csqa_ella_mind.CSQA_ELLAMIND_MC_HARD_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_HARD_FEWSHOT,
    )


def test_csqa_ellamind_cloze_easy_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        csqa_ella_mind.CSQA_ELLAMIND_CLOZE_EASY_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_EASY_ZEROSHOT,
    )

    assert_offline_oneshot_prompt(
        csqa_ella_mind.CSQA_ELLAMIND_CLOZE_EASY_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_EASY_FEWSHOT,
    )


def test_csqa_ellamind_cloze_hard_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        csqa_ella_mind.CSQA_ELLAMIND_CLOZE_HARD_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_HARD_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        csqa_ella_mind.CSQA_ELLAMIND_CLOZE_HARD_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_HARD_FEWSHOT,
    )


def test_csqa_ellamind_bpb_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        csqa_ella_mind.CSQA_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        csqa_ella_mind.CSQA_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )


# ---------------------------------------------------------------------------
# Distractor shuffling tests (offline)
# ---------------------------------------------------------------------------


def test_csqa_shuffling_is_deterministic_and_uses_expected_distractor_set() -> None:
    """Easy/hard variants should be deterministic and use different sources."""
    item: dict[str, Any] = {
        "question": "Was essen Pandas am liebsten?",
        "correct_answer": "Bambus",
        "easy_distractors": ["Pizza", "Eis", "Schokolade"],
        "hard_distractors": ["Blätter", "Gräser", "Kräuter"],
    }

    easy_task = csqa_ella_mind.CSQA_ELLAMIND_MC_EASY_DE(num_fewshot=0)
    hard_task = csqa_ella_mind.CSQA_ELLAMIND_MC_HARD_DE(num_fewshot=0)

    easy_choices_1, easy_idx_1 = easy_task._shuffled(item)
    easy_choices_2, easy_idx_2 = easy_task._shuffled(item)
    hard_choices, hard_idx = hard_task._shuffled(item)

    # Deterministic for identical input.
    assert (easy_choices_1, easy_idx_1) == (easy_choices_2, easy_idx_2)
    # Correct index should point to the correct answer.
    assert easy_choices_1[easy_idx_1] == "Bambus"
    assert hard_choices[hard_idx] == "Bambus"
    # Easy and hard variants must draw from different distractor pools.
    assert set(easy_choices_1) == {"Bambus", "Pizza", "Eis", "Schokolade"}
    assert set(hard_choices) == {"Bambus", "Blätter", "Gräser", "Kräuter"}
