"""Unit tests for eval_framework.tasks.formatting.

Covers: `format_mc_prompt`, `shuffle_correct_with_distractors`, `answer_key_to_index`,
`MCTaskMixin`, and `ClozeTaskMixin` (including `TRAILING_NEWLINE=False` for sentence-completion).
"""

import pytest

from eval_framework.tasks.base import Language, TaskFormat
from eval_framework.tasks.formatting import (
    ClozeTaskMixin,
    MCTaskMixin,
    answer_key_to_index,
    format_mc_prompt,
    shuffle_correct_with_distractors,
)


class TestFormatMcPrompt:
    def test_basic(self) -> None:
        result = format_mc_prompt("Question: What is 1+1?", ["1", "2", "3"])
        assert result == "Question: What is 1+1?\nA. 1\nB. 2\nC. 3\n"

    def test_space_prefixed_labels(self) -> None:
        result = format_mc_prompt("Question: What is 1+1?", ["1", "2", "3"], space_prefixed_labels=True)
        assert result == "Question: What is 1+1?\n A. 1\n B. 2\n C. 3\n"

    def test_two_choices(self) -> None:
        result = format_mc_prompt("Question: Is 1+1=2?", ["yes", "no"])
        assert result == "Question: Is 1+1=2?\nA. yes\nB. no\n"

    def test_trailing_newline(self) -> None:
        result = format_mc_prompt("Question: Is 1+1=2?", ["a"])
        assert result.endswith("\n")


class TestShuffleCorrectWithDistractors:
    def test_correct_index(self) -> None:
        choices, idx = shuffle_correct_with_distractors("cat", ["dog", "bird"], "seed")
        assert choices[idx] == "cat"

    def test_all_items_present(self) -> None:
        choices, _ = shuffle_correct_with_distractors("cat", ["dog", "bird"], "seed")
        assert sorted(choices) == sorted(["cat", "dog", "bird"])

    def test_deterministic(self) -> None:
        """Same arguments must always return the same result."""
        r1 = shuffle_correct_with_distractors("correct", ["d1", "d2", "d3"], "my question")
        r2 = shuffle_correct_with_distractors("correct", ["d1", "d2", "d3"], "my question")
        assert r1 == r2

    def test_different_seeds_may_differ(self) -> None:
        """Different seed texts should (almost certainly) produce different orders."""
        r1 = shuffle_correct_with_distractors("correct", ["d1", "d2", "d3"], "question A")
        r2 = shuffle_correct_with_distractors("correct", ["d1", "d2", "d3"], "question B")
        assert r1 != r2

    def test_single_distractor(self) -> None:
        choices, idx = shuffle_correct_with_distractors("right", ["wrong"], "seed")
        assert len(choices) == 2
        assert choices[idx] == "right"


class TestAnswerKeyToIndex:
    @pytest.mark.parametrize(
        "key,expected",
        [
            ("A", 0),
            ("B", 1),
            ("C", 2),
            ("D", 3),
            ("E", 4),
        ],
    )
    def test_letter_keys(self, key: str, expected: int) -> None:
        assert answer_key_to_index(key) == expected

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("1", 0),
            ("2", 1),
            ("3", 2),
            ("4", 3),
            ("5", 4),
        ],
    )
    def test_digit_keys(self, key: str, expected: int) -> None:
        assert answer_key_to_index(key) == expected


class _FakeBase:
    """Minimal stand-in for BaseTask so mixins can be tested without a real dataset."""

    def get_metadata(self) -> dict:
        return {"name": "FakeTask"}


class _FakeMCTask(MCTaskMixin, _FakeBase):
    def _get_raw_question(self, item: dict) -> str:
        return item["question"]

    def _get_choices(self, item: dict) -> list[str]:
        return item["choices"]

    def _get_correct_index(self, item: dict) -> int:
        return item["answer"]


_TEST_ITEM = {
    "question": "Capital of France?",
    "choices": ["Berlin", "Paris", "London"],
    "answer": 1,  # 0-based index of the correct answer
}


class TestMCTaskMixin:
    def setup_method(self) -> None:
        self.task = _FakeMCTask()

    def test_instruction_text(self) -> None:
        text = self.task._get_instruction_text(_TEST_ITEM)
        assert text == "Question: Capital of France?\nA. Berlin\nB. Paris\nC. London\n"

    def test_ground_truth(self) -> None:
        assert self.task._get_ground_truth(_TEST_ITEM) == " B"

    def test_possible_completions(self) -> None:
        assert self.task._get_possible_completions(_TEST_ITEM) == [" A", " B", " C"]

    def test_cue_text(self) -> None:
        assert self.task._get_cue_text(_TEST_ITEM) == "Answer:"

    def test_fewshot_target(self) -> None:
        assert self.task._get_fewshot_target_text(_TEST_ITEM) == "Answer: B"

    def test_metadata_includes_task_format(self) -> None:
        meta = self.task.get_metadata()
        assert meta["task_format"] == TaskFormat.MULTIPLE_CHOICE.value

    def test_space_prefixed_labels(self) -> None:
        class SpacedTask(_FakeMCTask):
            SPACE_PREFIXED_LABELS = True

        task = SpacedTask()
        text = task._get_instruction_text(_TEST_ITEM)
        assert " A. Berlin" in text

    def test_language_default_question_prefix(self) -> None:
        class GermanTask(_FakeMCTask):
            LANGUAGE = Language.DEU

        task = GermanTask()
        assert task._get_instruction_text(_TEST_ITEM).startswith("Frage: ")

    def test_custom_question_prefix(self) -> None:
        class OtherTask(_FakeMCTask):
            QUESTION_PREFIX = "Ziel: "

        task = OtherTask()
        assert task._get_instruction_text(_TEST_ITEM).startswith("Ziel: ")

    def test_question_prefix_on_base_class_not_overwritten(self) -> None:
        """QUESTION_PREFIX on a non-registered base must propagate to all variants
        and win over the mixin's own default despite the mixin coming first in the MRO."""

        class _GermanBase(_FakeBase):
            LANGUAGE = Language.DEU
            QUESTION_PREFIX = "Ziel: "  # differs from DEU registry default "Frage: "

        class TaskVariantA(MCTaskMixin, _GermanBase):
            pass

        class TaskVariantB(MCTaskMixin, _GermanBase):
            pass

        # Base-class value must win over both the mixin default and the registry default
        assert TaskVariantA.QUESTION_PREFIX == "Ziel: "
        assert TaskVariantB.QUESTION_PREFIX == "Ziel: "
        # CUE_TEXT still auto-filled from the registry (no explicit override)
        assert TaskVariantA.CUE_TEXT == "Antwort:"

    def test_override_question_text(self) -> None:
        """Tasks can override _get_question_text for non-standard formats."""

        class CustomTask(_FakeMCTask):
            def _get_question_text(self, item: dict) -> str:
                return f"Test: {item['question']} {item['choices'][0]}?"

        task = CustomTask()
        text = task._get_instruction_text(_TEST_ITEM)
        assert text.startswith("Test: Capital of France? Berlin?")


class _FakeClozeTask(ClozeTaskMixin, _FakeBase):
    def _get_raw_question(self, item: dict) -> str:
        return item["question"]

    def _get_choices(self, item: dict) -> list[str]:
        return item["choices"]

    def _get_correct_index(self, item: dict) -> int:
        return item["answer"]


class TestClozeTaskMixin:
    def setup_method(self) -> None:
        self.task = _FakeClozeTask()

    def test_instruction_text(self) -> None:
        # Default: question + trailing newline, no choices shown
        text = self.task._get_instruction_text(_TEST_ITEM)
        assert text == "Question: Capital of France?\n"
        assert "Berlin" not in text

    def test_ground_truth(self) -> None:
        assert self.task._get_ground_truth(_TEST_ITEM) == " Paris"

    def test_possible_completions(self) -> None:
        assert self.task._get_possible_completions(_TEST_ITEM) == [
            " Berlin",
            " Paris",
            " London",
        ]

    def test_cue_text(self) -> None:
        assert self.task._get_cue_text(_TEST_ITEM) == "Answer:"

    def test_fewshot_target(self) -> None:
        assert self.task._get_fewshot_target_text(_TEST_ITEM) == "Answer: Paris"

    def test_metadata_includes_task_format(self) -> None:
        meta = self.task.get_metadata()
        assert meta["task_format"] == TaskFormat.CLOZE.value

    def test_trailing_newline_false(self) -> None:
        """TRAILING_NEWLINE=False is used for sentence-completion tasks."""

        class SentenceTask(_FakeClozeTask):
            TRAILING_NEWLINE = False
            CUE_TEXT = ""

            def _get_question_text(self, item: dict) -> str:
                return item["fragment"]

        item = {
            "fragment": "The cat sat on the",
            "choices": ["mat", "floor"],
            "answer": 0,
        }
        task = SentenceTask()

        # No trailing newline
        assert task._get_instruction_text(item) == "The cat sat on the"
        # No cue text
        assert task._get_cue_text(item) == ""
        # Ground truth still space-prefixed
        assert task._get_ground_truth(item) == " mat"
