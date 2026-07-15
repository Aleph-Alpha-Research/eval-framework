from llm_sandbox.exceptions import SandboxTimeoutError

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion
from eval_framework.tasks.utils import run_python_code


class CodeCompletionAssertion(BaseMetric[Completion]):
    NAME = "Code Completion Accuracy"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        # this will always be a list, if return is "" this will be an empty list
        code = response.completion
        try:
            output = run_python_code(code, image="python:3.12-slim", runtime_configs={"mem_limit": "512m"})
        except SandboxTimeoutError:
            # The submitted code timed out (e.g. an infinite loop) -- a failing sample, not an infra
            # problem.
            import traceback

            return [
                MetricResult(
                    metric_name=self.NAME,
                    value=0.0,
                    higher_is_better=True,
                    code_execution_trace=traceback.format_exc(),
                )
            ]
        except Exception as e:
            # Any other sandbox/Docker error (e.g. an image pull rate limit) is an infra failure.
            return self._record_or_raise(e)

        # Split and filter out empty strings
        output_parts = [part for part in output.split() if part.strip()]

        if not output_parts:
            last_output = ""
        else:
            last_output = output_parts[-1]

        success = last_output == "True"
        return [
            MetricResult(
                metric_name=self.NAME,
                value=1.0 if success else 0.0,
                higher_is_better=True,
                error=None,
                code_execution_trace=output,
            )
        ]
