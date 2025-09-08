import traceback
from typing import Callable, Self

from pydantic import Field

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import BaseMetricContext, Completion, Error, extract_context_metric
from eval_framework.tasks.utils import CallableSerializer, ExecutionResult, execute_python_code_with_tests


class CodeExecutionBaseContext(BaseMetricContext):
    run_env: str = Field(description="Name of docker image to run unit-tests inside")
    code_prompt: str = Field(description="Prompt to LLM for code generation")
    test_code: str = Field(description="Python code that contains logic for unit test execution")
    benchmark_timeout: int = Field(default=60, description="Time in seconds for the full test execution run")
    package_downloads: dict[str, str | None] = Field(
        description="a dictionary listing the packages and their respective names in PyPiinto the LLM sandbox"
    )


class CodeExecutionPassAtOneContext(CodeExecutionBaseContext):
    snippet_merge_fn: str = Field(
        description="logic for merging LLM generated code with test execution code;"
        "this code will be passed into the sandbox to run the testing process"
        "This code is serialized"
    )
    output_parse_fn: str = Field(
        description="logic for parsing the output of test code execution run within the LLM sandbox"
        "This code is serialized"
    )


class RealtimeCodeExectionContext(CodeExecutionBaseContext):
    snippet_merge_fn: Callable[[str, str], str] = Field(
        description="logic for merging LLM generated code with test execution code;"
        "this code will be passed into the sandbox to run the testing process"
        "This code is deserialized"
    )
    output_parse_fn: Callable[[str], ExecutionResult] = Field(
        description="logic for parsing the output of test code execution run within the LLM sandbox"
        "This code is deserialized"
    )

    @classmethod
    def from_context(cls, context: CodeExecutionPassAtOneContext) -> Self:
        return cls(
            run_env=context.run_env,
            code_prompt=context.code_prompt,
            test_code=context.test_code,
            benchmark_timeout=context.benchmark_timeout,
            snippet_merge_fn=CallableSerializer.decode(context.snippet_merge_fn),
            output_parse_fn=CallableSerializer.decode(context.output_parse_fn),
            package_downloads=context.package_downloads,
        )


class CodeExecutionPassAtOne(BaseMetric[Completion]):
    NAME = "code-execution-pass@1"

    def __init__(self) -> None:
        self.k = 1
        # NOTE : this serializer should be the same class as initialized in the benchmark
        self.serializer = CallableSerializer()

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]
        try:
            context = extract_context_metric(response, CodeExecutionPassAtOneContext)
            parsed_context = RealtimeCodeExectionContext.from_context(context)
        except Exception as e:
            raise Exception(f"Failed to rebuild parsing functions => {e}")

        n = 1  # we only support N=1 at the moment
        try:
            c, output = self._count_correct_samples(response.completion, parsed_context)
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

    def _count_correct_samples(self, completion: str, context: RealtimeCodeExectionContext) -> tuple[int, str]:
        result = execute_python_code_with_tests(
            code=completion,
            test_code=context.test_code,
            package_mapping=context.package_downloads,
            merge_code_fn=context.snippet_merge_fn,
            image=context.run_env,
            timeout=context.benchmark_timeout,
            parse_output_fn=context.output_parse_fn,
        )
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
