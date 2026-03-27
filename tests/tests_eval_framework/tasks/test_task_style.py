"""Unit tests for eval_framework.tasks.task_style.

Covers: ``format_mc_prompt``, ``shuffle_correct_with_distractors``, ``answer_key_to_index``,
``MCStyle``, ``ClozeStyle``, and ``BaseTask`` task styler integration.
"""

import pytest

from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType, TaskStyle
from eval_framework.tasks.task_style import (
    ClozeStyle,
    MCStyle,
    answer_key_to_index,
    format_mc_prompt,
    shuffle_correct_with_distractors,
)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

_TEST_QUESTION = "Capital of France?"
_TEST_CHOICES = ["Berlin", "Paris", "London"]
_TEST_CORRECT_INDEX = 1  # Paris

_TEST_ITEM = {
    "question": _TEST_QUESTION,
    "choices": _TEST_CHOICES,
    "answer": _TEST_CORRECT_INDEX,
}


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MCStyle tests
# ---------------------------------------------------------------------------


class TestMCStyle:
    def setup_method(self) -> None:
        self.styler = MCStyle()

    def test_get_instruction_text(self) -> None:
        text = self.styler.get_instruction_text(_TEST_QUESTION, _TEST_CHOICES)
        assert text == "Question: Capital of France?\nA. Berlin\nB. Paris\nC. London\n"

    def test_get_ground_truth(self) -> None:
        assert self.styler.get_ground_truth(_TEST_CHOICES, _TEST_CORRECT_INDEX) == " B"

    def test_get_possible_completions(self) -> None:
        assert self.styler.get_possible_completions(_TEST_CHOICES) == [" A", " B", " C"]

    def test_get_cue_text(self) -> None:
        assert self.styler.get_cue_text() == "Answer:"

    def test_get_fewshot_target_text(self) -> None:
        assert self.styler.get_fewshot_target_text(_TEST_CHOICES, _TEST_CORRECT_INDEX) == "Answer: B"

    def test_extra_metadata(self) -> None:
        meta = self.styler.get_extra_metadata()
        assert meta["task_style"] == TaskStyle.MULTIPLE_CHOICE.value

    def test_response_type(self) -> None:
        assert self.styler.response_type == ResponseType.LOGLIKELIHOODS

    def test_space_prefixed_labels(self) -> None:
        styler = MCStyle(space_prefixed_labels=True)
        text = styler.get_instruction_text(_TEST_QUESTION, _TEST_CHOICES)
        assert " A. Berlin" in text

    def test_custom_question_prefix(self) -> None:
        styler = MCStyle(question_prefix="Ziel: ")
        text = styler.get_instruction_text(_TEST_QUESTION, _TEST_CHOICES)
        assert text.startswith("Ziel: ")

    def test_for_language_german(self) -> None:
        styler = MCStyle.for_language(Language.DEU)
        text = styler.get_instruction_text(_TEST_QUESTION, _TEST_CHOICES)
        assert text.startswith("Frage: ")
        assert styler.get_cue_text() == "Antwort:"

    def test_for_language_explicit_override(self) -> None:
        """Explicit kwargs take precedence over language defaults."""
        styler = MCStyle.for_language(Language.DEU, question_prefix="Ziel: ")
        text = styler.get_instruction_text(_TEST_QUESTION, _TEST_CHOICES)
        assert text.startswith("Ziel: ")
        assert styler.get_cue_text() == "Antwort:"

    def test_custom_get_question_text(self) -> None:
        """Subclassing the styler allows custom question formatting."""

        class CustomMCStyle(MCStyle):
            def get_question_text(self, raw_question: str) -> str:
                return f"Test: {raw_question} extra?"

        styler = CustomMCStyle()
        text = styler.get_instruction_text(_TEST_QUESTION, _TEST_CHOICES)
        assert text.startswith("Test: Capital of France? extra?")


# ---------------------------------------------------------------------------
# ClozeStyle tests
# ---------------------------------------------------------------------------


class TestClozeStyle:
    def setup_method(self) -> None:
        self.styler = ClozeStyle()

    def test_get_instruction_text(self) -> None:
        text = self.styler.get_instruction_text(_TEST_QUESTION, _TEST_CHOICES)
        assert text == "Question: Capital of France?\n"
        assert "Berlin" not in text

    def test_get_ground_truth(self) -> None:
        assert self.styler.get_ground_truth(_TEST_CHOICES, _TEST_CORRECT_INDEX) == " Paris"

    def test_get_possible_completions(self) -> None:
        assert self.styler.get_possible_completions(_TEST_CHOICES) == [
            " Berlin",
            " Paris",
            " London",
        ]

    def test_get_cue_text(self) -> None:
        assert self.styler.get_cue_text() == "Answer:"

    def test_get_fewshot_target_text(self) -> None:
        assert self.styler.get_fewshot_target_text(_TEST_CHOICES, _TEST_CORRECT_INDEX) == "Answer: Paris"

    def test_extra_metadata(self) -> None:
        meta = self.styler.get_extra_metadata()
        assert meta["task_style"] == TaskStyle.CLOZE.value

    def test_trailing_newline_false(self) -> None:
        """trailing_newline=False is used for sentence-completion tasks."""
        styler = ClozeStyle(question_prefix="", cue_text="", trailing_newline=False)
        fragment = "The cat sat on the"
        choices = ["mat", "floor"]

        assert styler.get_instruction_text(fragment, choices) == "The cat sat on the"
        assert styler.get_cue_text() == ""
        assert styler.get_ground_truth(choices, 0) == " mat"

    def test_for_language_german(self) -> None:
        styler = ClozeStyle.for_language(Language.DEU)
        text = styler.get_instruction_text(_TEST_QUESTION, _TEST_CHOICES)
        assert text.startswith("Frage: ")
        assert styler.get_cue_text() == "Antwort:"


# ---------------------------------------------------------------------------
# BaseTask task styler integration tests
# ---------------------------------------------------------------------------


class _ConcreteMCTask(BaseTask[str]):
    """Minimal concrete task for testing BaseTask with MCStyle."""

    NAME = "TestMCTask"
    DATASET_PATH = "test/dataset"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG
    TASK_STYLER = MCStyle()

    def _get_raw_question(self, item: dict) -> str:
        return item["question"]

    def _get_choices(self, item: dict) -> list[str]:
        return item["choices"]

    def _get_correct_index(self, item: dict) -> int:
        return item["answer"]


class _ConcreteClozeTask(BaseTask[str]):
    """Minimal concrete task for testing BaseTask with ClozeStyle."""

    NAME = "TestClozeTask"
    DATASET_PATH = "test/dataset"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG
    TASK_STYLER = ClozeStyle()

    def _get_raw_question(self, item: dict) -> str:
        return item["question"]

    def _get_choices(self, item: dict) -> list[str]:
        return item["choices"]

    def _get_correct_index(self, item: dict) -> int:
        return item["answer"]


class TestBaseTaskMCStyle:
    def setup_method(self) -> None:
        self.task = _ConcreteMCTask()

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

    def test_metadata_includes_task_style(self) -> None:
        meta = self.task.get_metadata()
        assert meta["task_style"] == TaskStyle.MULTIPLE_CHOICE.value
        assert meta["dataset_path"] == "test/dataset"

    def test_response_type_from_styler(self) -> None:
        assert self.task.TASK_STYLER.response_type == ResponseType.LOGLIKELIHOODS

    def test_metrics_from_styler(self) -> None:
        from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
            AccuracyLoglikelihood,
            AccuracyNormLoglikelihood,
        )
        from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood

        assert AccuracyLoglikelihood in self.task.TASK_STYLER.metrics
        assert AccuracyNormLoglikelihood in self.task.TASK_STYLER.metrics
        assert BitsPerByteLoglikelihood in self.task.TASK_STYLER.metrics


class TestBaseTaskClozeStyle:
    def setup_method(self) -> None:
        self.task = _ConcreteClozeTask()

    def test_instruction_text(self) -> None:
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

    def test_metadata_includes_task_style(self) -> None:
        meta = self.task.get_metadata()
        assert meta["task_style"] == TaskStyle.CLOZE.value


class TestBaseTaskStylerVariants:
    """Test task families sharing a base and swapping stylers."""

    def test_shared_base_mc_variant(self) -> None:
        class _Base(BaseTask[str]):
            NAME = "BaseTask"
            DATASET_PATH = "test/data"
            SAMPLE_SPLIT = "test"
            FEWSHOT_SPLIT = "train"
            SUBJECTS = [NO_SUBJECT]
            PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
            LANGUAGE = Language.ENG

            def _get_raw_question(self, item: dict) -> str:
                return item["question"]

            def _get_choices(self, item: dict) -> list[str]:
                return item["choices"]

            def _get_correct_index(self, item: dict) -> int:
                return item["answer"]

        class MCVariant(_Base):
            NAME = "MCVariant"
            TASK_STYLER = MCStyle()

        class ClozeVariant(_Base):
            NAME = "ClozeVariant"
            TASK_STYLER = ClozeStyle()

        mc_task = MCVariant()
        cloze_task = ClozeVariant()

        mc_text = mc_task._get_instruction_text(_TEST_ITEM)
        cloze_text = cloze_task._get_instruction_text(_TEST_ITEM)

        assert "A. Berlin" in mc_text
        assert "Berlin" not in cloze_text

        assert mc_task._get_ground_truth(_TEST_ITEM) == " B"
        assert cloze_task._get_ground_truth(_TEST_ITEM) == " Paris"

    def test_styler_override_in_subclass(self) -> None:
        """Subclass can swap styler without affecting parent."""

        class Parent(_ConcreteMCTask):
            NAME = "Parent"

        class Child(Parent):
            NAME = "Child"
            TASK_STYLER = ClozeStyle()

        parent = Parent()
        child = Child()

        assert parent._get_ground_truth(_TEST_ITEM) == " B"
        assert child._get_ground_truth(_TEST_ITEM) == " Paris"

    def test_metadata_response_type_from_styler(self) -> None:
        """get_metadata reads response_type from the styler, not from RESPONSE_TYPE."""
        task = _ConcreteMCTask()
        meta = task.get_metadata()
        assert meta["response_type"] == ResponseType.LOGLIKELIHOODS.value
