import pytest

from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.completion.niah_accuracy import NIAHAccuracy
from eval_framework.shared.types import Completion, Error, LanguageMetricContext
from template_formatting.formatter import Message, Role


class TestNIAHAccuracy:
    @pytest.fixture
    def metric(self) -> NIAHAccuracy:
        return NIAHAccuracy()

    @pytest.fixture
    def default_messages(self) -> list[Message]:
        return [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Please analyze this text and find the numbers."),
        ]

    def test_niah_single_correct(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test a correct answer for niah_single
        completion = Completion(
            id=1,
            subject="niah_test",
            prompt="Please analyze this text and find the numbers.",
            prompt_sequence_positions=100,
            completion="I found the number 12345 in the text.",
            ground_truth=["12345"],
            messages=default_messages,
            raw_completion="I found the number 12345 in the text.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="en"),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 1.0

    def test_niah_single_incorrect(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test an incorrect answer for niah_single
        completion = Completion(
            id=2,
            subject="niah_test",
            prompt="Please analyze this text and find the numbers.",
            prompt_sequence_positions=100,
            completion="I found the number 54321 in the text.",
            ground_truth=["12345"],
            messages=default_messages,
            raw_completion="I found the number 54321 in the text.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="en"),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 0.0

    def test_niah_single_none_word(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test when the model incorrectly says "none" for a task with an answer
        completion = Completion(
            id=3,
            subject="niah_test",
            prompt="Please analyze this text and find the numbers.",
            prompt_sequence_positions=100,
            completion="There is none to be found.",
            ground_truth=["12345"],
            messages=default_messages,
            raw_completion="There is none to be found.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="en"),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 0.0

    def test_niah_multi_correct(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test a correct answer for niah_multikey with multiple numbers
        completion = Completion(
            id=4,
            subject="niah_test",
            prompt="Please analyze this text and find the numbers.",
            prompt_sequence_positions=100,
            completion="I found the numbers 12345 and 67890 in the text.",
            ground_truth=["12345", "67890"],
            messages=default_messages,
            raw_completion="I found the numbers 12345 and 67890 in the text.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="en"),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 1.0

    def test_niah_multi_wrong_order(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test correct numbers but in different order (should still be correct)
        completion = Completion(
            id=5,
            subject="niah_test",
            prompt="Please analyze this text and find the numbers.",
            prompt_sequence_positions=100,
            completion="I found the numbers 67890 and 12345 in the text.",
            ground_truth=["12345", "67890"],
            messages=default_messages,
            raw_completion="I found the numbers 67890 and 12345 in the text.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="en"),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 1.0

    def test_niah_multi_missing_number(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test when one number is missing
        completion = Completion(
            id=6,
            subject="niah_test",
            prompt="Please analyze this text and find the numbers.",
            prompt_sequence_positions=100,
            completion="I found the number 12345 in the text.",
            ground_truth=["12345", "67890"],
            messages=default_messages,
            raw_completion="I found the number 12345 in the text.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="en"),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 0.0

    def test_niah_multi_extra_number(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test when there's an extra number
        completion = Completion(
            id=7,
            subject="niah_test",
            prompt="Please analyze this text and find the numbers.",
            prompt_sequence_positions=100,
            completion="I found the numbers 12345, 67890, and 11111 in the text.",
            ground_truth=["12345", "67890"],
            messages=default_messages,
            raw_completion="I found the numbers 12345, 67890, and 11111 in the text.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="en"),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 0.0

    def test_niah_none_correct(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test a correct "none" answer for niah_none
        completion = Completion(
            id=8,
            subject="niah_test",
            prompt="Please analyze this text and find the numbers.",
            prompt_sequence_positions=100,
            completion="There is none to be found in the text.",
            ground_truth=["none"],
            messages=default_messages,
            raw_completion="There is none to be found in the text.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="en"),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 1.0

    def test_niah_none_incorrect(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test an incorrect answer for niah_none
        completion = Completion(
            id=9,
            subject="niah_test",
            prompt="Please analyze this text and find the numbers.",
            prompt_sequence_positions=100,
            completion="I found the number 12345 in the text.",
            ground_truth=["none"],
            messages=default_messages,
            raw_completion="I found the number 12345 in the text.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="en"),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 0.0

    def test_niah_none_other_language(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test a correct "none" answer in another language
        completion = Completion(
            id=10,
            subject="niah_test",
            prompt="Veuillez analyser ce texte et trouver les nombres.",
            prompt_sequence_positions=100,
            completion="Aucun nombre n'a été trouvé dans le texte.",
            ground_truth=["none"],
            messages=[
                Message(role=Role.SYSTEM, content="Vous êtes un assistant utile."),
                Message(role=Role.USER, content="Veuillez analyser ce texte et trouver les nombres."),
            ],
            raw_completion="Aucun nombre n'a été trouvé dans le texte.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="fr"),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 1.0

    def test_niah_single_with_noise(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test with extra text and noise around the correct number
        completion = Completion(
            id=11,
            subject="niah_test",
            prompt="Please analyze this text and find the numbers.",
            prompt_sequence_positions=100,
            completion="After careful analysis, I believe the answer is 12345. There might be other numbers like 6 or "
            "7, but the multi-digit number is 12345.",
            ground_truth=["12345"],
            messages=default_messages,
            raw_completion="After careful analysis, I believe the answer is 12345. There might be other numbers like 6 "
            "or 7, but the multi-digit number is 12345.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="en"),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 1.0

    def test_niah_single_with_single_digits(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test with only single digits (should fail)
        completion = Completion(
            id=12,
            subject="niah_test",
            prompt="Please analyze this text and find the numbers.",
            prompt_sequence_positions=100,
            completion="I found the digits 1, 2, 3, 4, and 5.",
            ground_truth=["12345"],
            messages=default_messages,
            raw_completion="I found the digits 1, 2, 3, 4, and 5.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="en"),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value == 0.0

    def test_niah_with_error(self, metric: NIAHAccuracy, default_messages: list[Message]) -> None:
        # Test with an error in the completion
        completion = Completion(
            id=13,
            subject="niah_test",
            prompt="Please analyze this text and find the numbers.",
            prompt_sequence_positions=100,
            completion="I found the number 12345 in the text.",
            ground_truth=["12345"],
            messages=default_messages,
            raw_completion="I found the number 12345 in the text.",
            raw_completion_sequence_positions=50,
            context=LanguageMetricContext(language="en"),
            error=Error(error_class="TestError", message="Test error", traceback=""),
        )
        result: MetricResult = metric.calculate(completion)[0]
        assert result.value is None
        assert result.error is not None
        assert result.error.error_class == "TestError"


class TestNIAHAccuracyMethods:
    @pytest.fixture
    def metric(self) -> NIAHAccuracy:
        return NIAHAccuracy()

    # Tests for _compare_numbers method
    def test_compare_numbers_correct(self, metric: NIAHAccuracy) -> None:
        # Test with correct number
        result = metric._compare_numbers("en", ["12345"], "The answer is 12345.")
        assert result is True

    def test_compare_numbers_incorrect(self, metric: NIAHAccuracy) -> None:
        # Test with incorrect number
        result = metric._compare_numbers("en", ["12345"], "The answer is 54321.")
        assert result is False

    def test_compare_numbers_multiple_correct(self, metric: NIAHAccuracy) -> None:
        # Test with multiple correct numbers
        result = metric._compare_numbers("en", ["12345", "67890"], "Found numbers 12345 and 67890.")
        assert result is True

    def test_compare_numbers_multiple_correct_whitespace_separated(self, metric: NIAHAccuracy) -> None:
        # Test with multiple correct numbers
        result = metric._compare_numbers("en", ["12345", "67890"], "12345 67890")
        assert result is True

    def test_compare_numbers_multiple_correct_comma_separated(self, metric: NIAHAccuracy) -> None:
        # Test with multiple correct numbers
        result = metric._compare_numbers("en", ["12345", "67890"], "12345,67890.")
        assert result is True

    def test_compare_numbers_multiple_wrong_order(self, metric: NIAHAccuracy) -> None:
        # Test with multiple correct numbers in different order
        result = metric._compare_numbers("en", ["12345", "67890"], "Found numbers 67890 and 12345.")
        assert result is True

    def test_compare_numbers_missing_one(self, metric: NIAHAccuracy) -> None:
        # Test with one number missing
        result = metric._compare_numbers("en", ["12345", "67890"], "Found number 12345.")
        assert result is False

    def test_compare_numbers_extra_one(self, metric: NIAHAccuracy) -> None:
        # Test with an extra number
        result = metric._compare_numbers("en", ["12345", "67890"], "Found numbers 12345, 67890, and 11111.")
        assert result is False

    def test_compare_numbers_with_none_word(self, metric: NIAHAccuracy) -> None:
        # Test with "none" word (should fail)
        result = metric._compare_numbers("en", ["12345"], "There is none to be found.")
        assert result is False

    def test_compare_numbers_with_none_word_other_language(self, metric: NIAHAccuracy) -> None:
        # Test with "none" word in another language (should fail)
        result = metric._compare_numbers("fr", ["12345"], "Il n'y a aucun nombre.")
        assert result is False

    def test_compare_numbers_with_single_digits(self, metric: NIAHAccuracy) -> None:
        # Test with only single digits (should fail)
        result = metric._compare_numbers("en", ["12345"], "Found digits 1, 2, 3, 4, 5.")
        assert result is False

    def test_compare_numbers_empty_answer(self, metric: NIAHAccuracy) -> None:
        # Test with empty answer
        result = metric._compare_numbers("en", ["12345"], "")
        assert result is False

    def test_compare_numbers_non_numeric_answer(self, metric: NIAHAccuracy) -> None:
        # Test with non-numeric answer
        result = metric._compare_numbers("en", ["12345"], "The answer is twelve thousand three hundred forty-five.")
        assert result is False

    # Tests for _compare_none method
    def test_compare_none_correct_en(self, metric: NIAHAccuracy) -> None:
        # Test with correct "none" in English
        result = metric._compare_none("en", "There is none to be found.")
        assert result is True

    def test_compare_none_correct_fr(self, metric: NIAHAccuracy) -> None:
        # Test with correct "none" in French
        result = metric._compare_none("fr", "Il n'y a aucun nombre.")
        assert result is True

    def test_compare_none_correct_de(self, metric: NIAHAccuracy) -> None:
        # Test with correct "none" in German
        result = metric._compare_none("de", "Es gibt Keine vorhanden.")
        assert result is True

    def test_compare_none_correct_es(self, metric: NIAHAccuracy) -> None:
        # Test with correct "none" in Spanish
        result = metric._compare_none("es", "No hay ninguno.")
        assert result is True

    def test_compare_none_correct_zh(self, metric: NIAHAccuracy) -> None:
        # Test with correct "none" in Chinese
        result = metric._compare_none("zh", "没有数字，无。")
        assert result is True

    def test_compare_none_incorrect(self, metric: NIAHAccuracy) -> None:
        # Test with incorrect answer (has a number)
        result = metric._compare_none("en", "The answer is 12345.")
        assert result is False

    def test_compare_none_with_single_digit(self, metric: NIAHAccuracy) -> None:
        # Test with single digit (should be removed)
        result = metric._compare_none("en", "There is none. I see digit 5.")
        assert result is True

    def test_compare_none_with_multi_digit(self, metric: NIAHAccuracy) -> None:
        # Test with multi-digit number (should fail)
        result = metric._compare_none("en", "There is none. I see number 12345.")
        assert result is False

    def test_compare_none_empty_answer(self, metric: NIAHAccuracy) -> None:
        # Test with empty answer
        result = metric._compare_none("en", "")
        assert result is False

    def test_compare_none_wrong_language(self, metric: NIAHAccuracy) -> None:
        # Test with "none" word in wrong language
        result = metric._compare_none("fr", "There is none to be found.")
        assert result is False

    def test_compare_none_with_hyphenated_language(self, metric: NIAHAccuracy) -> None:
        # Test with hyphenated language code
        result = metric._compare_none("niah-fr", "Il n'y a aucun nombre.")
        assert result is True
