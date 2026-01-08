import pytest

from eval_framework.metrics.completion.math_reasoning_completion import MathReasoningCompletion
from eval_framework.shared.types import Completion


@pytest.fixture
def metric() -> MathReasoningCompletion:
    return MathReasoningCompletion()


def make_completion(completion: str, ground_truth: str | list[str]) -> Completion:
    """Helper to create Completion objects for testing."""
    return Completion(
        id=1,
        subject="test",
        ground_truth=ground_truth,
        prompt="test prompt",
        prompt_sequence_positions=None,
        messages=None,
        completion=completion,
        raw_completion=completion,
        raw_completion_sequence_positions=None,
    )


class TestNormalizeExpression:
    """Tests for the normalize_expression method - focus on non-trivial normalization."""

    @pytest.mark.parametrize(
        "input_expr,expected",
        [
            # Unit removal (at end only) - key normalization feature
            ("42 dollars", "42"),
            ("100 mph", "100"),
            ("30 degrees", "30"),
            # Formatting removal - key normalization feature
            (r"\text{hello}", "hello"),
            (r"\boxed{42}", "42"),
            # Dollar sign handling
            ("The answer is $42$", "42"),
            # Comma handling in numbers
            ("1,000", "1000"),
            ("1,000,000", "1000000"),
            ("1,23", "1,23"),  # Not valid thousands pattern - keep comma
            # Critical: does not break "infty" when removing "ft" unit
            ("\\infty", "\\infty"),
        ],
    )
    def test_normalize_expression(self, metric: MathReasoningCompletion, input_expr: str, expected: str) -> None:
        assert metric.normalize_expression(input_expr) == expected


class TestEquationSliding:
    """Tests for equation sliding logic in _is_str_correct."""

    @pytest.mark.parametrize(
        "str1,str2,expected",
        [
            # Response has more = signs - slides through response
            ("x=5", "5", True),
            ("a=b=5", "5", True),
            # Ground truth has more = signs - slides through ground truth
            ("5", "x=5", True),
            ("5", "a=b=5", True),
            # Equal = signs but different values
            ("x=5", "y=5", False),
            ("x=5", "x=6", False),
            # Mismatch after sliding
            ("x=5", "6", False),
        ],
    )
    def test_equation_sliding(self, metric: MathReasoningCompletion, str1: str, str2: str, expected: bool) -> None:
        assert metric._is_str_correct(str1, str2) == expected


class TestCalculateWithNormalization:
    """Tests that validate normalization is applied correctly to both response AND ground truth."""

    def test_units_stripped_from_response(self, metric: MathReasoningCompletion) -> None:
        """Response has units, ground truth doesn't."""
        response = make_completion("42 dollars", "42")
        results = metric.calculate(response)
        assert results[0].value == 1.0

    def test_units_stripped_from_ground_truth(self, metric: MathReasoningCompletion) -> None:
        """Ground truth has units, response doesn't - validates our fix."""
        response = make_completion("42", "42 dollars")
        results = metric.calculate(response)
        assert results[0].value == 1.0

    def test_formatting_stripped_from_both(self, metric: MathReasoningCompletion) -> None:
        """Both have formatting that gets normalized."""
        response = make_completion(r"\boxed{42}", r"\text{42}")
        results = metric.calculate(response)
        assert results[0].value == 1.0

    def test_comma_numbers_normalized(self, metric: MathReasoningCompletion) -> None:
        response = make_completion("1,000", "1000")
        results = metric.calculate(response)
        assert results[0].value == 1.0


class TestCalculateWithMultipleGroundTruths:
    """Tests that validate ground_truth_list is used correctly - validates our fix."""

    def test_matches_one_of_multiple_ground_truths(self, metric: MathReasoningCompletion) -> None:
        """Response matches one of several valid ground truths."""
        response = make_completion("42", ["40", "41", "42"])
        results = metric.calculate(response)
        assert results[0].value == 1.0

    def test_matches_none_of_multiple_ground_truths(self, metric: MathReasoningCompletion) -> None:
        """Response matches none of the ground truths."""
        response = make_completion("99", ["40", "41", "42"])
        results = metric.calculate(response)
        assert results[0].value == 0.0

    def test_normalization_applied_to_all_ground_truths(self, metric: MathReasoningCompletion) -> None:
        """Each ground truth in the list should be normalized."""
        response = make_completion("42", ["40 dollars", "41 dollars", "42 dollars"])
        results = metric.calculate(response)
        assert results[0].value == 1.0


class TestSymbolicEquivalence:
    """Tests for symbolic math equivalence - the core value of this metric."""

    @pytest.mark.parametrize(
        "response,ground_truth,expected",
        [
            # Numeric equivalence
            ("4", "2+2", 1.0),
            ("6", "2*3", 1.0),
            # Algebraic equivalence
            ("x^2", "x*x", 1.0),
            # Fraction equivalence
            (r"\frac{1}{2}", "0.5", 1.0),
            # No equivalence
            ("5", "4", 0.0),
        ],
    )
    def test_symbolic_equivalence(
        self, metric: MathReasoningCompletion, response: str, ground_truth: str, expected: float
    ) -> None:
        completion = make_completion(response, ground_truth)
        results = metric.calculate(completion)
        assert results[0].value == expected


class TestMetricMetadata:
    """Tests for metric result metadata."""

    def test_metric_name_and_properties(self, metric: MathReasoningCompletion) -> None:
        response = make_completion("42", "42")
        results = metric.calculate(response)
        assert len(results) == 1
        assert results[0].metric_name == "Math Reasoning Completion (symbolic)"
        assert results[0].higher_is_better is True
