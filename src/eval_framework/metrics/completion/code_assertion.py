from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion, Error
from eval_framework.tasks.utils import run_python_code


class CodeCompletionAssertion(BaseMetric[Completion]):
    NAME = "Code Completion Accuracy"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        # this will always be a list, if return is "" this will be an empty list
        code = response.completion
        output = run_python_code(code, image="python:3.12-slim")

        # Split and filter out empty strings
        output_parts = [part for part in output.split() if part.strip()]

        if not output_parts:
            last_output = ""
        else:
            last_output = output_parts[-1]

        success = last_output == "True"
        error = (
            None
            if success
            else Error(
                error_class="CodeCompletionAssertionError",
                message=f"Expected 'True' but got '{last_output}'",
                traceback=output,
            )
        )

        return [
            MetricResult(
                metric_name=self.NAME,
                value=1.0 if success else 0.0,
                higher_is_better=True,
                error=error,
                code_execution_trace=output,
            )
        ]
