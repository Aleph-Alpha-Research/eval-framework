from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from eval_framework.evaluation_generator import EvaluationGenerator
from eval_framework.metrics.base import MetricResult
from eval_framework.response_generator import ResponseGenerator
from eval_framework.result_processors.base import Result
from eval_framework.result_processors.result_processor import ResultsFileProcessor
from eval_framework.shared.types import Completion, Error, Loglikelihood
from eval_framework.task_names import TaskName
from eval_framework.tasks.eval_config import EvalConfig
from tests.conftest import MockLLM


class MockMetric:
    NAME = "MockMetric"

    def calculate(self, response: Completion | Loglikelihood) -> list[MetricResult]:
        return [MetricResult(metric_name="MockMetric", value=1.0, higher_is_better=True)]


def test_evaluator_run_completions(tmp_path: Path, should_preempt_callable: Callable) -> None:
    llm = MockLLM()

    config = EvalConfig(
        output_dir=tmp_path,
        num_fewshot=5,
        num_samples=2,
        task_name=TaskName.GPQA,
        llm_class=llm.__class__,
    )

    test_output_dir = tmp_path / "evaluator_test_output"

    file_processor = ResultsFileProcessor(test_output_dir)
    response_generator = ResponseGenerator(llm, config, file_processor)

    _ = response_generator.generate(should_preempt_callable)

    output_dir = Path(file_processor.output_dir)
    assert output_dir == test_output_dir
    assert (output_dir / "output.jsonl").exists()
    assert (output_dir / "metadata.json").exists()


def test_evaluator_run_eval(tmp_path: Path, should_preempt_callable: Callable) -> None:
    llm = MockLLM()
    config = EvalConfig(
        output_dir=tmp_path,
        num_fewshot=5,
        num_samples=2,
        task_name=TaskName.GPQA,
        llm_class=llm.__class__,
    )

    test_output_dir = tmp_path / "evaluator_test_output"

    file_processor = ResultsFileProcessor(test_output_dir)
    evaluator = EvaluationGenerator(config, file_processor)

    with pytest.raises(ValueError):
        evaluator.run_eval()

    response_generator = ResponseGenerator(llm, config, file_processor)
    _ = response_generator.generate(should_preempt_callable)

    _ = evaluator.run_eval()

    output_dir = Path(file_processor.output_dir)
    assert output_dir == test_output_dir
    assert (output_dir / "results.jsonl").exists()
    assert (output_dir / "aggregated_results.json").exists()


def test_evaluator_run_eval_no_completions(tmp_path: Path) -> None:
    llm = MockLLM()
    config = EvalConfig(
        output_dir=tmp_path,
        num_fewshot=5,
        num_samples=2,
        task_name=TaskName.GPQA,
        llm_class=llm.__class__,
    )

    test_output_dir = tmp_path / "evaluator_test_output"

    file_processor = ResultsFileProcessor(test_output_dir)
    evaluator = EvaluationGenerator(config, file_processor)

    with pytest.raises(ValueError) as exc_info:
        evaluator.run_eval()

    assert str(exc_info.value) == "No saved completions found. Run 'run_completions' first."


def test_evaluator_run_all(tmp_path: Path, should_preempt_callable: Callable) -> None:
    llm = MockLLM()
    config = EvalConfig(
        output_dir=tmp_path,
        num_fewshot=5,
        num_samples=2,
        task_name=TaskName.GPQA,
        llm_class=llm.__class__,
    )

    test_output_dir = tmp_path / "evaluator_test_output"

    file_processor = ResultsFileProcessor(test_output_dir)
    response_generator = ResponseGenerator(llm, config, file_processor)
    _ = response_generator.generate(should_preempt_callable)

    evaluator = EvaluationGenerator(config, file_processor)
    _ = evaluator.run_eval()

    output_dir = Path(file_processor.output_dir)
    assert output_dir == test_output_dir
    assert (output_dir / "output.jsonl").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "results.jsonl").exists()
    assert (output_dir / "aggregated_results.json").exists()


def test_aggregate_results(tmp_path: Path) -> None:
    llm = MockLLM()
    config = EvalConfig(
        output_dir=tmp_path,
        num_fewshot=5,
        num_samples=2,
        task_name=TaskName.GPQA,
        llm_class=llm.__class__,
    )
    evaluator = EvaluationGenerator(config, ResultsFileProcessor(tmp_path))

    responses = [
        ("subject1", "metric1", "key1", 5.0, None),
        ("subject1", "metric1", "key1", 2.0, None),
        (
            "subject1",
            "metric1",
            "key1",
            None,
            Error(error_class="AssertionError", message="asserted False!", traceback="just check the test data"),
        ),
        ("subject1", "metric1", "key1", 2.0, None),
        ("subject1", "metric1", "key2", 0.5, None),
        ("subject1", "metric1", "key2", 1.5, None),
        ("subject1", "metric1", "key2", None, None),
        ("subject1", "metric2", "key", 3.0, None),
        ("subject1", "metric2", "key", 3.0, None),
        ("subject1", "metric3", None, 20.0, None),
        ("subject2", "metric1", "key1", 4.0, None),
        ("subject2", "metric1", "key2", 0.0, None),
        (
            "subject2",
            "metric1",
            "key2",
            None,
            Error(error_class="AssertionError", message="asserted False!", traceback="just check the test data"),
        ),
        ("subject2", "metric2", "key", None, None),
        ("subject2", "metric3", None, 18.0, None),
    ]

    results = []
    for subject, metric_name, key, value, error in responses:
        results.append(
            Result(
                id=0,
                metric_name=metric_name,
                num_fewshot=0,
                key=key,
                subject=subject,
                llm_name="llm_name",
                task_name="task_name",
                metric_class_name=metric_name,
                value=value,
                higher_is_better=True,
                prompt="prompt",
                response="completion",
                error=error,
            )
        )

    aggregated_results = evaluator._aggregate_results(results)

    assert aggregated_results == {
        "Average metric1": 2.0,  # mean of all key-subject pairs (3.0, 4.0, 1.0, 0.0)
        "Average metric1 - key1": 3.5,  # mean of means of subjects (3.0 and 4.0)
        "Average metric1 - key2": 0.5,
        "Average metric1 - subject1": 2.0,  # mean of means of keys (3.0 and 1.0)
        "Average metric1 - subject2": 2.0,
        "Average metric2": 3.0,  # NaNs are skipped in the mean calculation
        "Average metric2 - subject1": 3.0,
        "Average metric2 - subject2": None,  # NaN appears
        # key in metric2 is not output because it's just a single submetric
        "Average metric3": 19.0,  # key=None case works
        "Average metric3 - subject1": 20.0,
        "Average metric3 - subject2": 18.0,
        "ErrorFreeRatio metric1": 0.8,
        "ErrorFreeRatio metric1 - key1": 0.8,
        "ErrorFreeRatio metric1 - key2": 0.8,
        "ErrorFreeRatio metric1 - subject1": 0.8571428571428571,
        "ErrorFreeRatio metric1 - subject2": 0.6666666666666666,
        "ErrorFreeRatio metric2": 1.0,
        "ErrorFreeRatio metric2 - subject1": 1.0,
        "ErrorFreeRatio metric2 - subject2": 1.0,
        "ErrorFreeRatio metric3": 1.0,
        "ErrorFreeRatio metric3 - subject1": 1.0,
        "ErrorFreeRatio metric3 - subject2": 1.0,
        "StdErr metric1 - key1": 0.8660254037844386,
        "NumSamples metric1 - key1": 4,
        "StdErr metric1 - key2": 0.3535533905932738,
        "NumSamples metric1 - key2": 4,
        "StdErr metric1 - subject1": 0.4978909578906802,
        "NumSamples metric1 - subject1": 6,
        "StdErr metric1 - subject2": None,  # NaN appear, cannot compute the std of only one value (division by N-1)
        "NumSamples metric1 - subject2": 2,
        "StdErr metric1": None,
        "NumSamples metric1": 16,
        "StdErr metric2 - subject1": 0,
        "NumSamples metric2 - subject1": 2,
        "StdErr metric2 - subject2": None,
        "NumSamples metric2 - subject2": 1,
        "StdErr metric2": None,
        "NumSamples metric2": 3,
        "StdErr metric3 - subject1": None,
        "NumSamples metric3 - subject1": 1,
        "StdErr metric3 - subject2": None,
        "NumSamples metric3 - subject2": 1,
        "StdErr metric3": None,
        "NumSamples metric3": 2,
    }


class TestFilterTaskSubjects:
    @pytest.fixture
    def response_generator(self) -> Any:
        # Create a mock instance with the real method
        mock_generator = MagicMock()
        mock_generator._filter_task_subjects = ResponseGenerator._filter_task_subjects.__get__(mock_generator)
        return mock_generator

    def test_no_task_subjects_specified(self, response_generator: Any) -> None:
        # Setup
        task_class = MagicMock()
        task_class.SUBJECTS = ["subject1", "subject2"]

        config = MagicMock(spec=EvalConfig)
        config.task_subjects = []
        # Create task_name as a MagicMock with a value attribute
        config.task_name = MagicMock()
        config.task_name.value = task_class

        response_generator.config = config

        # Execute
        result = response_generator._filter_task_subjects()

        # Assert
        assert result == task_class.SUBJECTS

    def test_filter_string_subjects(self, response_generator: Any) -> None:
        # Setup
        task_class = MagicMock()
        task_class.SUBJECTS = ["subject1", "subject2", "subject3"]
        task_class.NAME = "TestTask"

        config = MagicMock(spec=EvalConfig)
        config.task_subjects = ["subject1", "subject3"]
        config.task_name = MagicMock()
        config.task_name.value = task_class

        response_generator.config = config

        # Execute
        result = response_generator._filter_task_subjects()

        # Assert
        assert result == ["subject1", "subject3"]

    def test_filter_tuple_subjects(self, response_generator: Any) -> None:
        # Setup
        task_class = MagicMock()
        task_class.SUBJECTS = [("EN_US", "topic1"), ("EN_US", "topic2"), ("DE_DE", "topic1")]
        task_class.NAME = "TestTask"

        config = MagicMock(spec=EvalConfig)
        config.task_subjects = ["EN_US,topic1"]
        config.task_name = MagicMock()
        config.task_name.value = task_class

        response_generator.config = config

        # Execute
        result = response_generator._filter_task_subjects()

        # Assert
        assert result == [("EN_US", "topic1")]

    def test_filter_tuple_subjects_with_wildcard(self, response_generator: Any) -> None:
        # Setup
        task_class = MagicMock()
        task_class.SUBJECTS = [("EN_US", "topic1"), ("EN_US", "topic2"), ("DE_DE", "topic1")]
        task_class.NAME = "TestTask"

        config = MagicMock(spec=EvalConfig)
        config.task_subjects = ["EN_US,*"]
        config.task_name = MagicMock()
        config.task_name.value = task_class

        response_generator.config = config

        # Execute
        result = response_generator._filter_task_subjects()

        # Assert
        assert result == [("EN_US", "topic1"), ("EN_US", "topic2")]

    def test_filter_triple_tuple_subjects_with_wildcard(self, response_generator: Any) -> None:
        # Setup
        task_class = MagicMock()
        task_class.SUBJECTS = [
            ("EN_US", "topic1", "subtopic1"),
            ("EN_US", "topic1", "subtopic2"),
            ("EN_US", "topic2", "subtopic1"),
            ("DE_DE", "topic1", "subtopic1"),
        ]
        task_class.NAME = "TestTask"

        config = MagicMock(spec=EvalConfig)
        config.task_subjects = ["EN_US,topic1,*"]
        config.task_name = MagicMock()
        config.task_name.value = task_class

        response_generator.config = config

        # Execute
        result = response_generator._filter_task_subjects()

        # Assert
        assert result == [("EN_US", "topic1", "subtopic1"), ("EN_US", "topic1", "subtopic2")]

    def test_filter_triple_tuple_subjects_with_multiple_wildcards(self, response_generator: Any) -> None:
        # Setup
        task_class = MagicMock()
        task_class.SUBJECTS = [
            ("EN_US", "topic1", "subtopic1"),
            ("EN_US", "topic1", "subtopic2"),
            ("EN_US", "topic2", "subtopic1"),
            ("DE_DE", "topic1", "subtopic1"),
        ]
        task_class.NAME = "TestTask"

        config = MagicMock(spec=EvalConfig)
        config.task_subjects = ["*,topic1,*"]
        config.task_name = MagicMock()
        config.task_name.value = task_class

        response_generator.config = config

        # Execute
        result = response_generator._filter_task_subjects()

        # Assert
        assert result == [
            ("EN_US", "topic1", "subtopic1"),
            ("EN_US", "topic1", "subtopic2"),
            ("DE_DE", "topic1", "subtopic1"),
        ]

    def test_filter_tuple_subjects_multiple_filters(self, response_generator: Any) -> None:
        # Setup
        task_class = MagicMock()
        task_class.SUBJECTS = [("EN_US", "topic1"), ("EN_US", "topic2"), ("DE_DE", "topic1")]
        task_class.NAME = "TestTask"

        config = MagicMock(spec=EvalConfig)
        config.task_subjects = ["EN_US,topic1", "DE_DE,topic1"]
        config.task_name = MagicMock()
        config.task_name.value = task_class

        response_generator.config = config

        # Execute
        result = response_generator._filter_task_subjects()

        # Assert
        assert result == [("EN_US", "topic1"), ("DE_DE", "topic1")]

    def test_invalid_string_subject(self, response_generator: Any) -> None:
        # Setup
        task_class = MagicMock()
        task_class.SUBJECTS = ["subject1", "subject2"]
        task_class.NAME = "TestTask"

        config = MagicMock(spec=EvalConfig)
        config.task_subjects = ["invalid_subject"]
        config.task_name = MagicMock()
        config.task_name.value = task_class

        response_generator.config = config

        # Execute & Assert
        with pytest.raises(AssertionError):
            response_generator._filter_task_subjects()

    def test_invalid_tuple_subject_part(self, response_generator: Any) -> None:
        # Setup
        task_class = MagicMock()
        task_class.SUBJECTS = [("EN_US", "topic1"), ("EN_US", "topic2")]
        task_class.NAME = "TestTask"

        config = MagicMock(spec=EvalConfig)
        config.task_subjects = ["EN_US,invalid_topic"]
        config.task_name = MagicMock()
        config.task_name.value = task_class

        response_generator.config = config

        # Execute & Assert
        with pytest.raises(AssertionError):
            response_generator._filter_task_subjects()
