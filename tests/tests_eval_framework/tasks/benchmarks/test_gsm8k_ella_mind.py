"""Tests for the German GSM8K (EllaMind) tasks.

Dataset: https://huggingface.co/datasets/ellamind/gsm8k-platinum-multilingual (deu)

Tests:
- formatter hash test for every German GSM8K variant
- offline prompt assembly tests (generative GSM8K format and BPB format)
"""

from typing import Any

import pytest

import eval_framework.tasks.benchmarks.gsm8k_ella_mind as gsm8k_ella_mind
from eval_framework.tasks.benchmarks.gsm8k_ella_mind import GSM8K_Ellamind_DE_Platinum
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
from tests.tests_eval_framework.utils import DatasetPatcher

# ---------------------------------------------------------------------------
# Formatter hash tests (Hugging Face)
# ---------------------------------------------------------------------------


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", get_task_names_for_module("gsm8k_ella_mind"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "deu"

# Fictional rows following the GSM8K format. NOT real examples from the dataset.
_EVAL_ROW: dict[str, Any] = {
    "question": "Du hast 10 Äpfel und gibst 2 weg. Wie viele hast du noch?",
    "solution": "Wenn ich 10 Äpfel habe und 2 weggebe, habe ich 10 - 2 = 8 Äpfel übrig.",
    "final_answer": "8",
}

_FEWSHOT_ROW: dict[str, Any] = {
    "question": "Du hast 5 Orangen und gibst 3 weg. Wie viele hast du noch?",
    "solution": "Wenn ich 5 Orangen habe und 3 weggebe, habe ich 5 - 3 = 2 Orangen übrig.",
    "final_answer": "2",
}

# Expected prompts (messages, flat concat, ground truth, completions).
# --- GSM8K_Ellamind_DE_Platinum (generative) ---
_GSM8K_DE_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Du hast 10 Äpfel und gibst 2 weg. Wie viele hast du noch?\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Du hast 10 Äpfel und gibst 2 weg. Wie viele hast du noch?
Antwort:""",
    ground_truth="8",
    completions=None,
)

_GSM8K_DE_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Du hast 5 Orangen und gibst 3 weg. Wie viele hast du noch?\n",
        ),
        Message(
            role=Role.ASSISTANT,
            content="Antwort: Wenn ich 5 Orangen habe und 3 weggebe, habe ich 5 - 3 = 2 Orangen übrig. Daher ist die Antwort 2.",
        ),
        Message(
            role=Role.USER,
            content="Frage: Du hast 10 Äpfel und gibst 2 weg. Wie viele hast du noch?\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Du hast 5 Orangen und gibst 3 weg. Wie viele hast du noch?
Antwort: Wenn ich 5 Orangen habe und 3 weggebe, habe ich 5 - 3 = 2 Orangen übrig. Daher ist die Antwort 2.

Frage: Du hast 10 Äpfel und gibst 2 weg. Wie viele hast du noch?
Antwort:""",
    ground_truth=_GSM8K_DE_ZEROSHOT.ground_truth,
    completions=_GSM8K_DE_ZEROSHOT.completions,
)

# --- GSM8K_Ellamind_DE_BPB_Platinum ---
# The assembled prompt (and few-shot blocks) are identical to the generative task and therefore reused directly.
_BPB_GROUND_TRUTH = " Wenn ich 10 Äpfel habe und 2 weggebe, habe ich 10 - 2 = 8 Äpfel übrig. Daher ist die Antwort 8."

_BPB_ZEROSHOT = ExpectedPrompt(
    messages=_GSM8K_DE_ZEROSHOT.messages,
    concat=_GSM8K_DE_ZEROSHOT.concat,
    ground_truth=_BPB_GROUND_TRUTH,
    completions=[_BPB_GROUND_TRUTH],
)

_BPB_FEWSHOT = ExpectedPrompt(
    messages=_GSM8K_DE_FEWSHOT.messages,
    concat=_GSM8K_DE_FEWSHOT.concat,
    ground_truth=_BPB_GROUND_TRUTH,
    completions=[_BPB_GROUND_TRUTH],
)


# --- TESTS ---
def test_gsm8k_ella_mind_de_platinum_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        gsm8k_ella_mind.GSM8K_Ellamind_DE_Platinum,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_GSM8K_DE_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        gsm8k_ella_mind.GSM8K_Ellamind_DE_Platinum,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_GSM8K_DE_FEWSHOT,
    )


def test_gsm8k_ella_mind_de_bpb_platinum_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        gsm8k_ella_mind.GSM8K_Ellamind_DE_BPB_Platinum,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        gsm8k_ella_mind.GSM8K_Ellamind_DE_BPB_Platinum,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )


# ---------------------------------------------------------------------------
# Post-processing tests
# ---------------------------------------------------------------------------


def test_gsm8k_ella_mind_de_platinum_post_processing() -> None:
    input_output_pairs = [
        ("8", "8"),
        ("-8", "-8"),
        ("+8", "+8"),
        ("Die Antwort ist 8.", "8"),
        ("1000", "1000"),
        ("1,000", "1000"),
        ("1.000", "1000"),
        ("Die Antwort ist 1,200.", "1200"),
        (
            "Zuerst dachte ich die Antwort ist 0, aber dann habe ich gemerkt, dass die Antwort -8 ist.",
            "-8",
        ),
        ("Leider weiß ich die Antwort nicht.", "[invalid]"),
        ("", "[invalid]"),
    ]

    with DatasetPatcher[GSM8K_Ellamind_DE_Platinum](gsm8k_ella_mind.GSM8K_Ellamind_DE_Platinum) as task:
        for input_text, expected_output in input_output_pairs:
            assert task.post_process_generated_completion(input_text) == expected_output
