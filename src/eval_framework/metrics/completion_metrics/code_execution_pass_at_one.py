import os
import traceback

from eval_framework.logger import logger
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import BaseMetricContext, Completion, Error, extract_context_metric
from eval_framework.tasks.utils import BIG_CODE_BENCH_PACKAGE_MAPPING, execute_python_code_with_tests


class CodeExecutionPassAtOneContext(BaseMetricContext):
    code_prompt: str
    test_code: str


class CodeExecutionPassAtOne(BaseMetric[Completion]):
    NAME = "code-execution-pass@1"

    def __init__(self) -> None:
        self.k = 1
        # Get Docker image from environment variable
        self.python_image = os.environ.get("DOCKER_CODE_EXECUTION")
        if not self.python_image:
            raise ValueError(
                "Environment variable 'DOCKER_CODE_EXECUTION' must be set with a pre-built Docker image name. "
                "You can build the Docker image from the Dockerfile_codebench available in the repository's "
                "home directory."
            )

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        context = extract_context_metric(response, CodeExecutionPassAtOneContext)

        n = 1  # we only support N=1 at the moment
        try:
            c, output = self._count_correct_samples(response.completion, context.test_code)
        except Exception as e:
            error = Error(error_class=e.__class__.__name__, message=str(e), traceback=traceback.format_exc())
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=error)]

        pass_at_k_value = estimate_pass_at_k(n, c, self.k)
        return [
            MetricResult(
                metric_name=self.NAME,
                value=pass_at_k_value,
                higher_is_better=True,
                error=response.error,
                code_execution_trace=output,
            )
        ]

    def _count_correct_samples(self, completion: str, test_code: str) -> tuple[int, str]:
        result = execute_python_code_with_tests(
            completion, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING, image=self.python_image
        )
        logger.info(f"Output of code execution: {result.output}")
        return (1 if result.success else 0), result.output


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimates pass@k for a single problem.

    Parameters:
    n (int): Total number of generated samples.
    c (int): Number of correct samples.
    k (int): Number of attempts or samples considered.

    Returns:
    float: The pass@k value.
    """
    if n - c < k:
        return 1.0

    # Calculate the probability that at least one of the k samples is correct
    probability = 1.0
    for i in range(k):
        probability *= (n - c - i) / (n - i)

    return 1.0 - probability
