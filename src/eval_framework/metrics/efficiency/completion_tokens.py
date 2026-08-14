from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class NumCompletionTokens(BaseMetric[Completion]):
    """Number of tokens the model generated for the completion.

    Reads the token count backends already attach to the response, so no extra
    tokenisation is performed. Returns None when the backend did not report a
    count or when the sample errored.
    """

    NAME = "NumCompletionTokens"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error or response.raw_completion_num_tokens is None:
            value = None
        else:
            value = response.raw_completion_num_tokens

        return [
            MetricResult(
                metric_name=self.NAME,
                value=value,
                higher_is_better=False,
                error=response.error,
            )
        ]
