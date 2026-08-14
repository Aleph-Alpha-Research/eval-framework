from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class NumReasoningTokens(BaseMetric[Completion]):
    """Portion of the completion the model spent on reasoning (thinking).

    Reads the count backends attach to the response when the model exposes it
    (OpenAI reasoning models via `usage.completion_tokens_details.reasoning_tokens`;
    vLLM when started with `--reasoning-parser`). Non-reasoning models and
    backends that do not surface a per-response count return None.
    """

    NAME = "NumReasoningTokens"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error or response.raw_completion_reasoning_num_tokens is None:
            value = None
        else:
            value = response.raw_completion_reasoning_num_tokens

        return [
            MetricResult(
                metric_name=self.NAME,
                value=value,
                higher_is_better=False,
                error=response.error,
            )
        ]
