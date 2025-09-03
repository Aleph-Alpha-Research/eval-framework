import pytest

from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.completion.cwe_accuracy import CWEAccuracy
from eval_framework.shared.types import Completion, Error
from template_formatting.formatter import Message, Role


class TestCWEAccuracy:
    @pytest.fixture
    def metric(self) -> CWEAccuracy:
        return CWEAccuracy()

    @pytest.fixture
    def default_messages(self) -> list[Message]:
        return [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Please analyze this text and find the most common words."),
        ]

    def test_cwe_all_words_present(self, metric: CWEAccuracy, default_messages: list[Message]) -> None:
        # Test when all words are present in the answer
        completion = Completion(
            id=1,
            subject="cwe_test",
            prompt="Please analyze this text and find the most common words.",
            prompt_sequence_positions=100,
            completion="The most common words are: switch, efficacious, scrutiny, threat, obedient",
            ground_truth=["switch", "efficacious", "scrutiny", "threat", "obedient"],
            messages=default_messages,
            raw_completion="The most common words are: switch, efficacious, scrutiny, threat, obedient",
            raw_completion_sequence_positions=50,
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 1.0

    def test_cwe_all_words_different_order(self, metric: CWEAccuracy, default_messages: list[Message]) -> None:
        # Test when all words are present but in different order
        completion = Completion(
            id=2,
            subject="cwe_test",
            prompt="Please analyze this text and find the most common words.",
            prompt_sequence_positions=100,
            completion="The most common words are: threat, switch, obedient, scrutiny, efficacious",
            ground_truth=["switch", "efficacious", "scrutiny", "threat", "obedient"],
            messages=default_messages,
            raw_completion="The most common words are: threat, switch, obedient, scrutiny, efficacious",
            raw_completion_sequence_positions=50,
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 1.0

    def test_cwe_missing_word(self, metric: CWEAccuracy, default_messages: list[Message]) -> None:
        # Test when one word is missing
        completion = Completion(
            id=3,
            subject="cwe_test",
            prompt="Please analyze this text and find the most common words.",
            prompt_sequence_positions=100,
            completion="The most common words are: switch, efficacious, scrutiny, threat",
            ground_truth=["switch", "efficacious", "scrutiny", "threat", "obedient"],
            messages=default_messages,
            raw_completion="The most common words are: switch, efficacious, scrutiny, threat",
            raw_completion_sequence_positions=50,
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 0.0

    def test_cwe_extra_words(self, metric: CWEAccuracy, default_messages: list[Message]) -> None:
        # Test when there are extra words (should still pass if all required words are present)
        completion = Completion(
            id=4,
            subject="cwe_test",
            prompt="Please analyze this text and find the most common words.",
            prompt_sequence_positions=100,
            completion="The most common words are: switch, efficacious, scrutiny, threat, obedient, jumper, diabetes",
            ground_truth=["switch", "efficacious", "scrutiny", "threat", "obedient"],
            messages=default_messages,
            raw_completion="The most common words are: switch, efficacious, scrutiny, threat, obedient, jumper, "
            "diabetes",
            raw_completion_sequence_positions=50,
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 1.0

    def test_cwe_case_insensitive(self, metric: CWEAccuracy, default_messages: list[Message]) -> None:
        # Test case insensitivity
        completion = Completion(
            id=5,
            subject="cwe_test",
            prompt="Please analyze this text and find the most common words.",
            prompt_sequence_positions=100,
            completion="The most common words are: SWITCH, Efficacious, sCrUtInY, Threat, OBEDIENT",
            ground_truth=["switch", "efficacious", "scrutiny", "threat", "obedient"],
            messages=default_messages,
            raw_completion="The most common words are: SWITCH, Efficacious, sCrUtInY, Threat, OBEDIENT",
            raw_completion_sequence_positions=50,
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 1.0

    def test_cwe_with_explanation(self, metric: CWEAccuracy, default_messages: list[Message]) -> None:
        # Test with additional explanation text
        completion = Completion(
            id=6,
            subject="cwe_test",
            prompt="Please analyze this text and find the most common words.",
            prompt_sequence_positions=100,
            completion="After analyzing the text, I found that the most common words are switch, efficacious, "
            "scrutiny, threat, and obedient. These words appeared multiple times throughout the document.",
            ground_truth=["switch", "efficacious", "scrutiny", "threat", "obedient"],
            messages=default_messages,
            raw_completion="After analyzing the text, I found that the most common words are switch, efficacious, "
            "scrutiny, threat, and obedient. These words appeared multiple times throughout the document.",
            raw_completion_sequence_positions=50,
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 1.0

    def test_cwe_with_error(self, metric: CWEAccuracy, default_messages: list[Message]) -> None:
        # Test with an error in the completion
        completion = Completion(
            id=7,
            subject="cwe_test",
            prompt="Please analyze this text and find the most common words.",
            prompt_sequence_positions=100,
            completion="The most common words are: switch, efficacious, scrutiny, threat, obedient",
            ground_truth=["switch", "efficacious", "scrutiny", "threat", "obedient"],
            messages=default_messages,
            raw_completion="The most common words are: switch, efficacious, scrutiny, threat, obedient",
            raw_completion_sequence_positions=50,
            error=Error(error_class="TestError", message="Test error", traceback=""),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value is None
        assert result.error is not None
        assert result.error.error_class == "TestError"


class TestCWEAccuracyMethods:
    @pytest.fixture
    def metric(self) -> CWEAccuracy:
        return CWEAccuracy()

    # Tests for _is_correct_answer method
    def test_is_correct_answer_all_present(self, metric: CWEAccuracy) -> None:
        # Test when all words are present
        result = metric._is_answer_correct(
            ["switch", "efficacious", "scrutiny", "threat", "obedient"],
            "The most common words are: switch, efficacious, scrutiny, threat, obedient",
        )
        assert result is True

    def test_is_correct_answer_different_order(self, metric: CWEAccuracy) -> None:
        # Test when all words are present but in different order
        result = metric._is_answer_correct(
            ["switch", "efficacious", "scrutiny", "threat", "obedient"],
            "The most common words are: threat, switch, obedient, scrutiny, efficacious",
        )
        assert result is True

    def test_is_correct_answer_missing_word(self, metric: CWEAccuracy) -> None:
        # Test when one word is missing
        result = metric._is_answer_correct(
            ["switch", "efficacious", "scrutiny", "threat", "obedient"],
            "The most common words are: switch, efficacious, scrutiny, threat",
        )
        assert result is False

    def test_is_correct_answer_extra_words(self, metric: CWEAccuracy) -> None:
        # Test when there are extra words
        result = metric._is_answer_correct(
            ["switch", "efficacious", "scrutiny", "threat", "obedient"],
            "The most common words are: switch, efficacious, scrutiny, threat, obedient, jumper, diabetes",
        )
        assert result is True

    def test_is_correct_answer_case_insensitive(self, metric: CWEAccuracy) -> None:
        # Test case insensitivity
        result = metric._is_answer_correct(
            ["switch", "efficacious", "scrutiny", "threat", "obedient"],
            "The most common words are: SWITCH, Efficacious, sCrUtInY, Threat, OBEDIENT",
        )
        assert result is True

    def test_is_correct_answer_with_explanation(self, metric: CWEAccuracy) -> None:
        # Test with additional explanation text
        result = metric._is_answer_correct(
            ["switch", "efficacious", "scrutiny", "threat", "obedient"],
            "After analyzing the text, I found that the most common words are switch, efficacious, "
            "scrutiny, threat, and obedient. These words appeared multiple times throughout the document.",
        )
        assert result is True

    def test_is_correct_answer_partial_word_match(self, metric: CWEAccuracy) -> None:
        # Test with partial word matches (should fail)
        result = metric._is_answer_correct(
            ["switch", "efficacious"], "The most common words are: switching, efficaciousness"
        )
        assert result is False

    def test_is_correct_answer_empty_answer(self, metric: CWEAccuracy) -> None:
        # Test with empty answer
        result = metric._is_answer_correct(["switch", "efficacious", "scrutiny", "threat", "obedient"], "")
        assert result is False

    def test_is_correct_answer_empty_ground_truth(self, metric: CWEAccuracy) -> None:
        # Test with empty ground truth
        result = metric._is_answer_correct(
            [], "The most common words are: switch, efficacious, scrutiny, threat, obedient"
        )
        assert result is True

    def test_is_correct_answer_with_punctuation(self, metric: CWEAccuracy) -> None:
        # Test with punctuation around words
        result = metric._is_answer_correct(
            ["switch", "efficacious", "scrutiny", "threat", "obedient"],
            "The most common words are: 'switch', 'efficacious', 'scrutiny', 'threat', and 'obedient'.",
        )
        assert result is True
