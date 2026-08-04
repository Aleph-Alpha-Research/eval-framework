"""Tests for the German Winogrande (EllaMind) tasks.

Tests:
- formatter hash test for every Winogrande variant
- offline prompt assembly tests for the Cloze and MC formats
- partial-eval sample construction tests (offline)
"""

from typing import Any

import pytest

import eval_framework.tasks.benchmarks.winogrande_ellamind as winogrande_ellamind
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_winogrande_ellamind_tasks
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

# Registry for this test suite only holding winogrande_ellamind tasks
_winogrande_ellamind_registry = Registry()
register_winogrande_ellamind_tasks(registry=_winogrande_ellamind_registry)

# ---------------------------------------------------------------------------
# Formatter hash tests (Hugging Face)
# ---------------------------------------------------------------------------


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _winogrande_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_winogrande_ellamind_registry)


# ---------------------------------------------------------------------------
# Offline prompt assembly tests (use fictional dataset)
# ---------------------------------------------------------------------------

_SUBJECT = "deu"


# Fictional rows. NOT real examples from the Winogrande dataset.
# The blank ``_`` is filled by ``option1`` or ``option2``; ``answer`` selects the correct one.
_EVAL_ROW: dict[str, Any] = {
    "sentence": "Georg war größer als Johannes, daher konnte _ weiter sehen.",
    "option1": "Georg",
    "option2": "Johannes",
    "answer": "1",  # option1 (Georg) is correct
}

_FEWSHOT_ROW: dict[str, Any] = {
    "sentence": "Anna lief schneller als Maria, daher gewann _ das Rennen.",
    "option1": "Maria",
    "option2": "Anna",
    "answer": "2",  # option2 (Anna) is correct
}

# Expected prompts (messages, flat concat, ground truth, completions).
# --- WINOGRANDE_ELLAMIND_CLOZE_DE ---
_CLOZE_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Georg war größer als Johannes, daher konnte"),
    ],
    concat="Georg war größer als Johannes, daher konnte",
    ground_truth=" Georg weiter sehen.",
    completions=[" Georg weiter sehen.", " Johannes weiter sehen."],
)

_CLOZE_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Anna lief schneller als Maria, daher gewann"),
        Message(role=Role.ASSISTANT, content=" Anna das Rennen."),
        Message(role=Role.USER, content="Georg war größer als Johannes, daher konnte"),
    ],
    concat="""\
Anna lief schneller als Maria, daher gewann Anna das Rennen.

Georg war größer als Johannes, daher konnte""",
    ground_truth=_CLOZE_ZEROSHOT.ground_truth,
    completions=_CLOZE_ZEROSHOT.completions,
)

# --- WINOGRANDE_ELLAMIND_MC_DE ---
_MC_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Georg war größer als Johannes, daher konnte\nA. Georg weiter sehen.\nB. Johannes weiter sehen.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Georg war größer als Johannes, daher konnte
A. Georg weiter sehen.
B. Johannes weiter sehen.
Antwort:""",
    ground_truth=" A",
    completions=[" A", " B"],
)

_MC_FEWSHOT = ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Anna lief schneller als Maria, daher gewann\nA. Maria das Rennen.\nB. Anna das Rennen.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort: B"),
        Message(
            role=Role.USER,
            content="Frage: Georg war größer als Johannes, daher konnte\nA. Georg weiter sehen.\nB. Johannes weiter sehen.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    concat="""\
Frage: Anna lief schneller als Maria, daher gewann
A. Maria das Rennen.
B. Anna das Rennen.
Antwort: B

Frage: Georg war größer als Johannes, daher konnte
A. Georg weiter sehen.
B. Johannes weiter sehen.
Antwort:""",
    ground_truth=_MC_ZEROSHOT.ground_truth,
    completions=_MC_ZEROSHOT.completions,
)


# --- TESTS ---
def test_winogrande_ellamind_cloze_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        winogrande_ellamind.WINOGRANDE_ELLAMIND_CLOZE_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        winogrande_ellamind.WINOGRANDE_ELLAMIND_CLOZE_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_CLOZE_FEWSHOT,
    )


def test_winogrande_ellamind_mc_de_offline_prompt_formatting() -> None:
    assert_offline_zeroshot_prompt(
        winogrande_ellamind.WINOGRANDE_ELLAMIND_MC_DE,
        eval_row=_EVAL_ROW,
        subjects=[_SUBJECT],
        expected=_MC_ZEROSHOT,
    )
    assert_offline_oneshot_prompt(
        winogrande_ellamind.WINOGRANDE_ELLAMIND_MC_DE,
        eval_row=_EVAL_ROW,
        fewshot_row=_FEWSHOT_ROW,
        subjects=[_SUBJECT],
        expected=_MC_FEWSHOT,
    )


# ---------------------------------------------------------------------------
# Partial-eval sample construction tests (offline)
# ---------------------------------------------------------------------------


def test_winogrande_partial_eval_create_samples_has_expected_pair_structure() -> None:
    """One dataset item must become exactly two paired samples.

    This test documents the core "partial evaluation" contract:
    we evaluate the same suffix under two different prompts ("prefix + option").
    """
    task = winogrande_ellamind.WINOGRANDE_ELLAMIND_PARTIAL_EVAL_DE(num_fewshot=0)
    samples = task._create_samples(_EVAL_ROW, index=10, subject="deu")

    assert len(samples) == 2
    assert [sample.id for sample in samples] == [10, 11]
    assert [sample.ground_truth for sample in samples].count("True") == 1
    assert [sample.ground_truth for sample in samples].count("False") == 1
    assert all(sample.possible_completions == [" weiter sehen."] for sample in samples)
    assert samples[0].messages[-1].content == "Georg war größer als Johannes, daher konnte Georg"
    assert samples[1].messages[-1].content == "Georg war größer als Johannes, daher konnte Johannes"


def test_winogrande_partial_eval_iterate_samples_uses_consecutive_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """`iterate_samples` increments IDs per yielded sample, not per dataset item.

    This is the key behavior that makes `id=index + opt_idx` work for
    partial-eval tasks (both upstream Winogrande and EllaMind variant).
    """

    def fake_load_dataset(self: winogrande_ellamind.WINOGRANDE_ELLAMIND_PARTIAL_EVAL_DE, subject: str) -> None:
        _ = subject  # Subject value is irrelevant for this unit test.
        self.dataset = {self.SAMPLE_SPLIT: [_EVAL_ROW, _FEWSHOT_ROW]}

    monkeypatch.setattr(winogrande_ellamind.WINOGRANDE_ELLAMIND_PARTIAL_EVAL_DE, "_load_dataset", fake_load_dataset)
    task = winogrande_ellamind.WINOGRANDE_ELLAMIND_PARTIAL_EVAL_DE(num_fewshot=0)
    sample_ids = [sample.id for sample in task.iterate_samples(num_samples=4)]

    assert sample_ids == [0, 1, 2, 3]
    # PartialEvalAccuracy groups samples by id // 2, so pairs should map to
    # item ids [0, 0, 1, 1].
    assert [sample_id // 2 for sample_id in sample_ids] == [0, 0, 1, 1]
