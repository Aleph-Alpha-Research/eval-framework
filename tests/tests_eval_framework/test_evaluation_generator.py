from collections.abc import Callable
from pathlib import Path

import pytest
import wandb

from eval_framework.evaluation_generator import EvaluationGenerator
from eval_framework.metrics.aggregators.aggregators import IdentifierMean, PassAtK
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.response_generator import ResponseGenerator
from eval_framework.result_processors.base import Result
from eval_framework.result_processors.result_processor import ResultsFileProcessor
from eval_framework.shared.types import Completion, Error, Loglikelihood
from eval_framework.tasks.benchmarks.gpqa import GPQA
from eval_framework.tasks.eval_config import EvalConfig
from tests.tests_eval_framework.conftest import MockLLM


class MockMetric:
    NAME = "MockMetric"

    def calculate(self, response: Completion | Loglikelihood) -> list[MetricResult]:
        return [MetricResult(metric_name="MockMetric", value=1.0, higher_is_better=True)]


class MockPassAtKMetric(BaseMetric):
    NAME = "ExactMatch"
    AGGREGATORS = [PassAtK(k=1), PassAtK(k=2)]

    def calculate(self, response: Completion | Loglikelihood) -> list[MetricResult]:
        return []


class MockIdentifierMeanMetric(BaseMetric):
    NAME = "ExactMatch"
    AGGREGATORS = [IdentifierMean()]

    def calculate(self, response: Completion | Loglikelihood) -> list[MetricResult]:
        return []


def test_evaluator_run_completions(tmp_path: Path, should_preempt_callable: Callable) -> None:
    llm = MockLLM()

    config = EvalConfig(
        output_dir=tmp_path,
        num_fewshot=5,
        num_samples=2,
        task_name=GPQA.NAME,
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


def test_evaluator_run_eval(tmp_path: Path, should_preempt_callable: Callable, wandb_run: wandb.Run) -> None:
    llm = MockLLM()
    config = EvalConfig(
        output_dir=tmp_path,
        num_fewshot=5,
        num_samples=2,
        task_name=GPQA.NAME,
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


def test_evaluator_run_eval_no_completions(tmp_path: Path, wandb_run: wandb.Run) -> None:
    llm = MockLLM()
    config = EvalConfig(
        output_dir=tmp_path,
        num_fewshot=5,
        num_samples=2,
        task_name=GPQA.NAME,
        llm_class=llm.__class__,
    )

    test_output_dir = tmp_path / "evaluator_test_output"

    file_processor = ResultsFileProcessor(test_output_dir)
    evaluator = EvaluationGenerator(config, file_processor)

    with pytest.raises(ValueError) as exc_info:
        evaluator.run_eval()

    assert str(exc_info.value) == "No saved completions found. Run 'run_completions' first."


def test_evaluator_run_all(tmp_path: Path, should_preempt_callable: Callable, wandb_run: wandb.Run) -> None:
    llm = MockLLM()
    config = EvalConfig(
        output_dir=tmp_path,
        num_fewshot=5,
        num_samples=2,
        task_name=GPQA.NAME,
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
        task_name=GPQA.NAME,
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
        "Average metric1 (including Errors)": 1.8125,  # errors treated as 0: (2.25, 1.0, 4.0, 0.0)
        "Average metric1 - key1": 3.5,  # mean of means of subjects (3.0 and 4.0)
        "Average metric1 (including Errors) - key1": 3.125,  # errors as 0: (2.25, 4.0)
        "Average metric1 - key2": 0.5,
        "Average metric1 (including Errors) - key2": 0.5,  # errors as 0: (1.0, 0.0)
        "Average metric1 - subject1": 2.0,  # mean of means of keys (3.0 and 1.0)
        "Average metric1 (including Errors) - subject1": 1.625,  # errors as 0: (2.25, 1.0)
        "Average metric1 - subject2": 2.0,
        "Average metric1 (including Errors) - subject2": 2.0,  # errors as 0: (4.0, 0.0)
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


def test_aggregate_results_with_aggregators(tmp_path: Path) -> None:
    llm = MockLLM()
    config = EvalConfig(
        output_dir=tmp_path,
        num_fewshot=0,
        num_samples=2,
        task_name=GPQA.NAME,
        llm_class=llm.__class__,
    )
    evaluator = EvaluationGenerator(config, ResultsFileProcessor(tmp_path))
    evaluator.metrics = [MockPassAtKMetric]

    responses = [
        ("subject1", "ExactMatch", "key1", "p1", 1.0, None),
        ("subject1", "ExactMatch", "key1", "p1", 1.0, None),
        ("subject1", "ExactMatch", "key1", "p1", 0.0, None),
        ("subject1", "ExactMatch", "key1", "p2", 0.0, None),
        ("subject1", "ExactMatch", "key2", "p3", 1.0, None),
        ("subject2", "ExactMatch", "key1", "p4", 1.0, None),
        ("subject2", "ExactMatch", "key2", "p5", 0.0, None),
        (
            "subject2",
            "ExactMatch",
            "key1",
            "p6",
            None,
            Error(error_class="AssertionError", message="asserted False!", traceback="just check the test data"),
        ),
    ]

    results = []
    for subject, metric_name, key, prompt, value, error in responses:
        results.append(
            Result(
                id=0,
                metric_name=metric_name,
                num_fewshot=0,
                key=key,
                subject=subject,
                llm_name="llm_name",
                task_name="task_name",
                metric_class_name="MockPassAtKMetric",
                value=value,
                higher_is_better=True,
                prompt=prompt,
                response="completion",
                error=error,
            )
        )

    aggregated = evaluator._aggregate_results_with_aggregators(results)
    # The pass@1 DF that gets aggregated is:
    # p1 s1 k1 2/3
    # p2 s1 k1 0
    # p3 s1 k2 1
    # p4 s2 k1 1
    # p5 s2 k2 0

    # The pass@2 DF that gets aggregated is:
    # p1 s1 k1 1
    # p2 s1 k1 0
    # p3 s1 k2 1
    # p4 s2 k1 1
    # p5 s2 k2 0
    assert aggregated == {
        "Pass@1 MockPassAtKMetric.ExactMatch": pytest.approx(((2 / 3 + 0) / 2 + 1 + 1 + 0) / 4),
        "Pass@1 ExactMatch - key1 - subject1": pytest.approx((2 / 3 + 0) / 2),
        "Pass@1 ExactMatch - key1 - subject2": 1,
        "Pass@1 ExactMatch - key2 - subject1": 1,
        "Pass@1 ExactMatch - key2 - subject2": 0.0,
        "Pass@2 MockPassAtKMetric.ExactMatch": pytest.approx((1 / 2 + 1 + 1 + 0) / 4),
        "Pass@2 ExactMatch - key1 - subject1": pytest.approx((1 + 0) / 2),
        "Pass@2 ExactMatch - key1 - subject2": 1,
        "Pass@2 ExactMatch - key2 - subject1": 1,
        "Pass@2 ExactMatch - key2 - subject2": 0.0,
    }


def test_aggregate_results_with_identifier_mean(tmp_path: Path) -> None:
    llm = MockLLM()
    config = EvalConfig(
        output_dir=tmp_path,
        num_fewshot=0,
        num_samples=2,
        task_name=GPQA.NAME,
        llm_class=llm.__class__,
    )
    evaluator = EvaluationGenerator(config, ResultsFileProcessor(tmp_path))
    evaluator.metrics = [MockIdentifierMeanMetric]

    responses = [
        ("subject1", "ExactMatch", "key1", "p1", 1.0, None),
        ("subject1", "ExactMatch", "key1", "p1", 1.0, None),
        ("subject1", "ExactMatch", "key1", "p1", 0.0, None),
        ("subject1", "ExactMatch", "key1", "p2", 0.0, None),
        ("subject1", "ExactMatch", "key2", "p3", 1.0, None),
        ("subject2", "ExactMatch", "key1", "p4", 1.0, None),
        ("subject2", "ExactMatch", "key2", "p5", 0.0, None),
        (
            "subject2",
            "ExactMatch",
            "key1",
            "p5",
            None,
            Error(error_class="AssertionError", message="asserted False!", traceback="just check the test data"),
        ),
    ]

    results = []
    for subject, metric_name, key, prompt, value, error in responses:
        results.append(
            Result(
                id=0,
                metric_name=metric_name,
                num_fewshot=0,
                key=key,
                subject=subject,
                llm_name="llm_name",
                task_name="task_name",
                metric_class_name="MockIdentifierMeanMetric",
                value=value,
                higher_is_better=True,
                prompt=prompt,
                response="completion",
                error=error,
            )
        )

    aggregated = evaluator._aggregate_results_with_aggregators(results)

    # The aggregator with IdentifierMean DF that gets aggregated is:
    # p1 s1 k1 2/3
    # p2 s1 k1 0
    # p3 s1 k2 1
    # p4 s2 k1 1
    # p5 s2 k2 0
    #
    # First loop: groupby (key, subject) means → (1/3, 1.0, 1.0, 0) → overall mean = 0.5833
    assert aggregated == {
        "IdentifierMean MockIdentifierMeanMetric.ExactMatch": pytest.approx(((2 / 3 + 0) / 2 + 1 + 1 + 0) / 4),
        "IdentifierMean ExactMatch - key1 - subject1": pytest.approx((2 / 3 + 0) / 2),
        "IdentifierMean ExactMatch - key2 - subject1": 1,
        "IdentifierMean ExactMatch - key1 - subject2": 1,
        "IdentifierMean ExactMatch - key2 - subject2": 0,
    }
