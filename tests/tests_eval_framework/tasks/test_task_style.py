"""Unit tests for eval_framework.tasks.task_style.

Covers: ``format_mc_prompt``, ``shuffle_correct_with_distractors``, ``answer_key_to_index``,
and the stylers (``MCStyle``, ``ClozeStyle``, ``MCCompletionStyle``, ``BPBStyle``).
"""

import pytest

from eval_framework.contract import ResponseType
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.tasks import task_style
from eval_framework.tasks.base import Language, TaskStyle
from eval_framework.tasks.task_style import (
    BPBStyle,
    ClozeStyle,
    MCCompletionStyle,
    MCStyle,
    answer_key_to_index,
    format_mc_prompt,
    question_cue_text,
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
# MCCompletionStyle tests
# ---------------------------------------------------------------------------


class TestMCCompletionStyle:
    def setup_method(self) -> None:
        self.styler = MCCompletionStyle()

    def test_get_instruction_text(self) -> None:
        text = self.styler.get_instruction_text(_TEST_QUESTION, _TEST_CHOICES)
        assert text == "Question: Capital of France?\nA. Berlin\nB. Paris\nC. London\n"

    def test_get_ground_truth(self) -> None:
        assert self.styler.get_ground_truth(_TEST_CHOICES, _TEST_CORRECT_INDEX) == "Paris"

    def test_get_fewshot_target_text(self) -> None:
        assert self.styler.get_fewshot_target_text(_TEST_CHOICES, _TEST_CORRECT_INDEX) == "Answer: Paris"

    def test_get_possible_completions_none(self) -> None:
        assert self.styler.get_possible_completions(_TEST_CHOICES) is None

    def test_get_cue_text(self) -> None:
        assert self.styler.get_cue_text() == "Answer:"

    def test_response_type(self) -> None:
        assert self.styler.response_type == ResponseType.COMPLETION

    def test_extra_metadata(self) -> None:
        meta = self.styler.get_extra_metadata()
        assert meta["task_style"] == TaskStyle.MULTIPLE_CHOICE.value


# ---------------------------------------------------------------------------
# BPBStyle tests
# ---------------------------------------------------------------------------


class TestBPBStyle:
    def setup_method(self) -> None:
        self.styler = BPBStyle()

    def test_get_instruction_text(self) -> None:
        """Prompt is identical to ClozeStyle — no choices shown."""
        text = self.styler.get_instruction_text(_TEST_QUESTION, _TEST_CHOICES)
        assert text == "Question: Capital of France?\n"
        assert "Berlin" not in text

    def test_get_ground_truth(self) -> None:
        assert self.styler.get_ground_truth(_TEST_CHOICES, _TEST_CORRECT_INDEX) == " Paris"

    def test_get_possible_completions_returns_only_ground_truth(self) -> None:
        completions = self.styler.get_possible_completions(_TEST_CHOICES, _TEST_CORRECT_INDEX)
        assert completions == [" Paris"]
        assert len(completions) == 1

    def test_get_cue_text(self) -> None:
        assert self.styler.get_cue_text() == "Answer:"

    def test_get_fewshot_target_text(self) -> None:
        assert self.styler.get_fewshot_target_text(_TEST_CHOICES, _TEST_CORRECT_INDEX) == "Answer: Paris"

    def test_extra_metadata(self) -> None:
        meta = self.styler.get_extra_metadata()
        assert meta["task_style"] == TaskStyle.BPB.value

    def test_response_type(self) -> None:
        assert self.styler.response_type == ResponseType.LOGLIKELIHOODS

    def test_metrics_bpb_only(self) -> None:
        from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import AccuracyLoglikelihood
        from eval_framework.metrics.loglikelihood.bpb_variants import BitsPerByteVariantsLoglikelihood

        assert self.styler.metrics == [BitsPerByteLoglikelihood, BitsPerByteVariantsLoglikelihood]
        assert AccuracyLoglikelihood not in self.styler.metrics

    def test_for_language_german(self) -> None:
        styler = BPBStyle.for_language(Language.DEU)
        text = styler.get_instruction_text(_TEST_QUESTION, _TEST_CHOICES)
        assert text.startswith("Frage: ")
        assert styler.get_cue_text() == "Antwort:"


@pytest.fixture
def swe_without_defaults(monkeypatch: pytest.MonkeyPatch) -> Language:
    """Swedish with no language defaults, whether or not the real table gains them later."""
    monkeypatch.delitem(task_style._DEFAULT_QUESTION_CUE_TEXT, Language.SWE, raising=False)
    return Language.SWE


def test_question_cue_text_unknown_language_raises(swe_without_defaults: Language) -> None:
    with pytest.raises(ValueError, match="SWE"):
        question_cue_text(swe_without_defaults)


def test_for_language_unknown_language_keeps_styler_defaults(swe_without_defaults: Language) -> None:
    """The lenient counterpart: a styler built for a language without defaults keeps its own (English) ones."""
    styler = ClozeStyle.for_language(swe_without_defaults)
    assert styler.question_prefix == "Question: "
    assert styler.get_cue_text() == "Answer:"
