"""Tests for the German SIQA (EllaMind) tasks.

Tests:
- formatter hash test for every SIQA variant
- offline prompt assembly tests
"""

from typing import Any

import pytest

import eval_framework.tasks.benchmarks.siqa_ellamind as siqa_ellamind
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_siqa_ellamind_tasks
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

# Registry for this test suite only holding siqa_ellamind tasks
_siqa_ellamind_registry = Registry()
register_siqa_ellamind_tasks(registry=_siqa_ellamind_registry)

# ---------------------------------------------------------------------------
# Formatter hash tests (Hugging Face)
# ---------------------------------------------------------------------------


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _siqa_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_siqa_ellamind_registry)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "deu"

# Fictional rows following the SIQA format. NOT real examples from the SIQA dataset.
# Option order and correct letters are shuffled deterministically (seed: question+answer).
_EVAL_ROW: dict[str, Any] = {
    "context": "Max kommt nach Hause und findet sein Zimmer aufgeräumt.",
    "question": "Was hat Max als nächstes getan?",
    "correct_answer": "Er hat sich bedankt.",
    "easy_distractors": ["Er ist wütend geworden.", "Er ist gegangen."],
    "hard_distractors": ["Er hat gegessen.", "Er hat geschlafen."],
}

_FEWSHOT_ROW: dict[str, Any] = {
    "context": "Lisa hat ihr Buch vergessen.",
    "question": "Wie hat Lisa sich gefühlt?",
    "correct_answer": "Ärgerlich.",
    "easy_distractors": ["Glücklich.", "Müde."],
    "hard_distractors": ["Neugierig.", "Aufgeregt."],
}

# Expected prompts (messages, flat concat, ground truth, completions).
# --- SIQA_ELLAMIND_MC_EASY_DE ---
_MC_EASY_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?\nA. Er ist wütend geworden.\nB. Er hat sich bedankt.\nC. Er ist gegangen.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?
A. Er ist wütend geworden.
B. Er hat sich bedankt.
C. Er ist gegangen.
Antwort:""",
    ground_truth=" B",
    completions=[" A", " B", " C"],
)

_MC_EASY_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Lisa hat ihr Buch vergessen. Wie hat Lisa sich gefühlt?\nA. Ärgerlich.\nB. Müde.\nC. Glücklich.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort: A"),
        Message(
            role=Role.USER,
            content="Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?\nA. Er ist wütend geworden.\nB. Er hat sich bedankt.\nC. Er ist gegangen.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Lisa hat ihr Buch vergessen. Wie hat Lisa sich gefühlt?
A. Ärgerlich.
B. Müde.
C. Glücklich.
Antwort: A

Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?
A. Er ist wütend geworden.
B. Er hat sich bedankt.
C. Er ist gegangen.
Antwort:""",
    ground_truth=_MC_EASY_ZEROSHOT.ground_truth,
    completions=_MC_EASY_ZEROSHOT.completions,
)

# --- SIQA_ELLAMIND_MC_HARD_DE ---
_MC_HARD_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?\nA. Er hat gegessen.\nB. Er hat sich bedankt.\nC. Er hat geschlafen.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?
A. Er hat gegessen.
B. Er hat sich bedankt.
C. Er hat geschlafen.
Antwort:""",
    ground_truth=" B",
    completions=[" A", " B", " C"],
)

_MC_HARD_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Lisa hat ihr Buch vergessen. Wie hat Lisa sich gefühlt?\nA. Ärgerlich.\nB. Aufgeregt.\nC. Neugierig.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort: A"),
        Message(
            role=Role.USER,
            content="Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?\nA. Er hat gegessen.\nB. Er hat sich bedankt.\nC. Er hat geschlafen.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Lisa hat ihr Buch vergessen. Wie hat Lisa sich gefühlt?
A. Ärgerlich.
B. Aufgeregt.
C. Neugierig.
Antwort: A

Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?
A. Er hat gegessen.
B. Er hat sich bedankt.
C. Er hat geschlafen.
Antwort:""",
    ground_truth=_MC_HARD_ZEROSHOT.ground_truth,
    completions=_MC_HARD_ZEROSHOT.completions,
)

# --- SIQA_ELLAMIND_CLOZE_EASY_DE ---
# Cloze prompts show no options, so the easy/hard variants share the same prompt; only the
# scored completions differ.
_CLOZE_EASY_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?
Antwort:""",
    ground_truth=" Er hat sich bedankt.",
    completions=[
        " Er ist wütend geworden.",
        " Er hat sich bedankt.",
        " Er ist gegangen.",
    ],
)

_CLOZE_EASY_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Lisa hat ihr Buch vergessen. Wie hat Lisa sich gefühlt?\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort: Ärgerlich."),
        Message(
            role=Role.USER,
            content="Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Lisa hat ihr Buch vergessen. Wie hat Lisa sich gefühlt?
Antwort: Ärgerlich.

Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?
Antwort:""",
    ground_truth=_CLOZE_EASY_ZEROSHOT.ground_truth,
    completions=_CLOZE_EASY_ZEROSHOT.completions,
)

# --- SIQA_ELLAMIND_CLOZE_HARD_DE ---
# Same prompt as cloze-easy; only the (hard) distractor completions differ.
_CLOZE_HARD_ZEROSHOT = ExpectedPrompt(
    messages=_CLOZE_EASY_ZEROSHOT.messages,
    concat=_CLOZE_EASY_ZEROSHOT.concat,
    ground_truth=_CLOZE_EASY_ZEROSHOT.ground_truth,
    completions=[
        " Er hat gegessen.",
        " Er hat sich bedankt.",
        " Er hat geschlafen.",
    ],
)

_CLOZE_HARD_FEWSHOT = ExpectedPrompt(
    messages=_CLOZE_EASY_FEWSHOT.messages,
    concat=_CLOZE_EASY_FEWSHOT.concat,
    ground_truth=_CLOZE_EASY_ZEROSHOT.ground_truth,
    completions=_CLOZE_HARD_ZEROSHOT.completions,
)

# --- SIQA_ELLAMIND_BPB_DE ---
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
def test_siqa_ellamind_mc_easy_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        siqa_ellamind.SIQA_ELLAMIND_MC_EASY_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_EASY_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        siqa_ellamind.SIQA_ELLAMIND_MC_EASY_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_EASY_FEWSHOT,
    )


def test_siqa_ellamind_mc_hard_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        siqa_ellamind.SIQA_ELLAMIND_MC_HARD_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_HARD_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        siqa_ellamind.SIQA_ELLAMIND_MC_HARD_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_HARD_FEWSHOT,
    )


def test_siqa_ellamind_cloze_easy_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        siqa_ellamind.SIQA_ELLAMIND_CLOZE_EASY_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_EASY_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        siqa_ellamind.SIQA_ELLAMIND_CLOZE_EASY_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_EASY_FEWSHOT,
    )


def test_siqa_ellamind_cloze_hard_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        siqa_ellamind.SIQA_ELLAMIND_CLOZE_HARD_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_HARD_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        siqa_ellamind.SIQA_ELLAMIND_CLOZE_HARD_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_HARD_FEWSHOT,
    )


def test_siqa_ellamind_bpb_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        siqa_ellamind.SIQA_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        siqa_ellamind.SIQA_ELLAMIND_BPB_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_BPB_FEWSHOT,
    )
