from unittest.mock import Mock

import pytest

from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm.base import BaseLLMJudgeMetric, safe_metric_calculation
from eval_framework.shared.types import Completion, Error
from template_formatting.formatter import Message, Role


def create_test_completion(
    with_error: bool = False,
    error_message: str = "Test error",
) -> Completion:
    """Create a test Completion object with optional error."""
    return Completion(
        id=0,
        subject="test_subject",
        ground_truth="expected answer",
        prompt="test prompt",
        prompt_sequence_positions=None,
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="What is 2+2?"),
        ],
        completion="The answer is 4.",
        raw_completion="The answer is 4.",
        raw_completion_sequence_positions=None,
        error=Error(error_class="TestError", message=error_message, traceback="") if with_error else None,
    )


@pytest.fixture
def completion() -> Completion:
    """Fixture providing a default test completion without errors."""
    return create_test_completion()


@pytest.fixture
def completion_with_error() -> Completion:
    """Fixture providing a test completion with a pre-existing error."""
    return create_test_completion(with_error=True)


class SingleMetricJudge(BaseLLMJudgeMetric):
    """Test metric class with a single metric name."""

    NAME = "Test Single Metric"

    def __init__(self, llm_judge: BaseLLM, should_raise: bool = False):
        super().__init__(llm_judge)
        self._should_raise = should_raise
        self._call_count = 0

    @safe_metric_calculation
    def calculate(self, response: Completion) -> list[MetricResult]:
        self._call_count += 1
        if self._should_raise:
            raise ValueError("Simulated LLM judge failure")
        return [
            MetricResult(
                metric_name=self.NAME,
                value=1.0,
                higher_is_better=True,
                llm_judge_prompt="test prompt",
                llm_judge_response="test response",
            )
        ]


class MultiMetricJudge(BaseLLMJudgeMetric):
    """Test metric class with multiple metric names (like LLMJudgeInstruction)."""

    NAME = "Test Multi Metric"
    KEYS = ["quality", "accuracy", "relevance"]

    def __init__(self, llm_judge: BaseLLM, should_raise: bool = False):
        super().__init__(llm_judge)
        self._should_raise = should_raise

    @safe_metric_calculation
    def calculate(self, response: Completion) -> list[MetricResult]:
        if self._should_raise:
            raise RuntimeError("Simulated multi-metric failure")
        return [
            MetricResult(
                metric_name=f"{self.NAME}/{key}",
                value=0.8,
                higher_is_better=True,
            )
            for key in self.KEYS
        ]


class LowerIsBetterMetric(BaseLLMJudgeMetric):
    """Test metric where lower values are better (like WorldKnowledge)."""

    NAME = "Test Lower Is Better"
    _higher_is_better = False

    def __init__(self, llm_judge: BaseLLM, should_raise: bool = False):
        super().__init__(llm_judge)
        self._should_raise = should_raise

    @safe_metric_calculation
    def calculate(self, response: Completion) -> list[MetricResult]:
        if self._should_raise:
            raise Exception("Simulated failure")
        return [
            MetricResult(
                metric_name=self.NAME,
                value=0.1,
                higher_is_better=False,
            )
        ]


class TestSuccessfulExecution:
    """Tests verifying the decorator doesn't interfere with normal operation."""

    def test_single_metric_success(self, completion: Completion) -> None:
        """Decorator should not interfere with successful single metric calculation."""
        llm = Mock(spec=BaseLLM)
        metric = SingleMetricJudge(llm, should_raise=False)

        results = metric.calculate(completion)

        assert len(results) == 1
        assert results[0].metric_name == "Test Single Metric"
        assert results[0].value == 1.0
        assert results[0].error is None

    def test_multi_metric_success(self, completion: Completion) -> None:
        """Decorator should not interfere with successful multi-metric calculation."""
        llm = Mock(spec=BaseLLM)
        metric = MultiMetricJudge(llm, should_raise=False)

        results = metric.calculate(completion)

        assert len(results) == 3
        assert all(r.error is None for r in results)
        assert {r.metric_name for r in results} == {
            "Test Multi Metric/quality",
            "Test Multi Metric/accuracy",
            "Test Multi Metric/relevance",
        }


class TestPreExistingResponseError:
    """Tests for handling responses that already have errors."""

    def test_single_metric_with_response_error(self) -> None:
        """Should return error result without calling the actual calculate logic."""
        llm = Mock(spec=BaseLLM)
        metric = SingleMetricJudge(llm, should_raise=False)
        response = create_test_completion(with_error=True, error_message="Pre-existing error")

        results = metric.calculate(response)

        assert len(results) == 1
        assert results[0].metric_name == "Test Single Metric"
        assert results[0].value is None
        assert results[0].error is not None
        assert results[0].error.message == "Pre-existing error"
        # Verify the actual calculate logic was never called
        assert metric._call_count == 0

    def test_multi_metric_with_response_error(self, completion_with_error: Completion) -> None:
        """Should return error results for ALL metric names when response has error."""
        llm = Mock(spec=BaseLLM)
        metric = MultiMetricJudge(llm, should_raise=False)

        results = metric.calculate(completion_with_error)

        # Should return one error result per metric name
        assert len(results) == 3
        assert all(r.value is None for r in results)
        assert all(r.error is not None for r in results)
        assert {r.metric_name for r in results} == {
            "Test Multi Metric/quality",
            "Test Multi Metric/accuracy",
            "Test Multi Metric/relevance",
        }

    def test_higher_is_better_preserved_for_response_error(self, completion_with_error: Completion) -> None:
        """Error results should preserve the higher_is_better setting."""
        llm = Mock(spec=BaseLLM)
        metric = LowerIsBetterMetric(llm, should_raise=False)

        results = metric.calculate(completion_with_error)

        assert len(results) == 1
        assert results[0].higher_is_better is False


class TestExceptionHandling:
    """Tests for catching exceptions during metric calculation."""

    def test_single_metric_exception_caught(self, completion: Completion) -> None:
        """Should catch exception and return error result instead of crashing."""
        llm = Mock(spec=BaseLLM)
        metric = SingleMetricJudge(llm, should_raise=True)

        # Should NOT raise - exception should be caught
        results = metric.calculate(completion)

        assert len(results) == 1
        assert results[0].value is None
        assert results[0].error is not None
        assert results[0].error.error_class == "ValueError"
        assert "Simulated LLM judge failure" in results[0].error.message
        # Verify traceback is captured for debugging
        assert results[0].error.traceback != ""
        assert "ValueError" in results[0].error.traceback
        assert "Simulated LLM judge failure" in results[0].error.traceback

    def test_multi_metric_exception_caught(self, completion: Completion) -> None:
        """Should return error results for ALL metric names when exception occurs."""
        llm = Mock(spec=BaseLLM)
        metric = MultiMetricJudge(llm, should_raise=True)

        results = metric.calculate(completion)

        assert len(results) == 3
        assert all(r.value is None for r in results)
        assert all(r.error is not None for r in results)
        assert all(r.error.error_class == "RuntimeError" for r in results)

    def test_higher_is_better_preserved_for_exception(self, completion: Completion) -> None:
        """Error results from exceptions should preserve higher_is_better setting."""
        llm = Mock(spec=BaseLLM)
        metric = LowerIsBetterMetric(llm, should_raise=True)

        results = metric.calculate(completion)

        assert len(results) == 1
        assert results[0].higher_is_better is False


class TestIntegrationWithRealMetrics:
    """Integration tests using actual metric classes from the codebase."""

    def test_conciseness_with_exception(self, completion: Completion) -> None:
        """Test LLMJudgeConciseness handles exceptions gracefully."""
        from eval_framework.metrics.llm.llm_judge_conciseness import LLMJudgeConciseness

        llm = Mock(spec=BaseLLM)
        # Simulate LLM throwing an exception
        llm.generate_from_messages.side_effect = ConnectionError("API timeout")

        metric = LLMJudgeConciseness(llm)

        # Should NOT crash
        results = metric.calculate(completion)

        assert len(results) == 1
        assert results[0].value is None
        assert results[0].error is not None
        assert results[0].error.error_class == "ConnectionError"

    def test_instruction_with_response_error(self) -> None:
        """Test LLMJudgeInstruction handles response errors correctly."""
        from eval_framework.metrics.llm.llm_judge_instruction import LLMJudgeInstruction

        llm = Mock(spec=BaseLLM)
        metric = LLMJudgeInstruction(llm)
        response = create_test_completion(with_error=True, error_message="Model error")

        results = metric.calculate(response)

        # Should return error for all 7 metric keys
        assert len(results) == 7
        assert all(r.error is not None for r in results)
        assert all(r.value is None for r in results)

    def test_world_knowledge_preserves_higher_is_better(self, completion: Completion) -> None:
        """Test LLMJudgeWorldKnowledge error results have higher_is_better=False."""
        from eval_framework.metrics.llm.llm_judge_world_knowledge import LLMJudgeWorldKnowledge

        llm = Mock(spec=BaseLLM)
        llm.generate_from_messages.side_effect = Exception("Test exception")

        metric = LLMJudgeWorldKnowledge(llm)

        results = metric.calculate(completion)

        assert len(results) == 1
        assert results[0].higher_is_better is False
        assert results[0].error is not None
