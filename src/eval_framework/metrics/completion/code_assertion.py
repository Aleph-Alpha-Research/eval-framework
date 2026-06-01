from llm_sandbox.exceptions import SandboxTimeoutError

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
        try:
            output = run_python_code(code, image="python:3.12-slim")
        except SandboxTimeoutError as e:
            # The submitted code timed out (e.g. an infinite loop) -- a failing sample, not an
            # infra problem. Any other sandbox/Docker error (e.g. an image pull rate limit) is left
            # to propagate so the run fails instead of being scored as a wrong answer.
            import traceback

            return [
                MetricResult(
                    metric_name=self.NAME,
                    value=0.0,
                    higher_is_better=True,
                    error=Error(error_class=e.__class__.__name__, message=str(e), traceback=traceback.format_exc()),
                )
            ]

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
