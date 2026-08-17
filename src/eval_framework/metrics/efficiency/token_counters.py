from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class TokenCounts(BaseMetric[Completion]):
    """Number of tokens the model generated for the completion, and how many of
    those were spent on reasoning (thinking).

    Reads the token counts backends already attach to the response, so no extra
    tokenisation is performed. Each value independently falls back to None when
    the backend did not report that particular count or when the sample errored.
    Reasoning counts are only exposed by some backends (e.g. OpenAI reasoning
    models via `usage.completion_tokens_details.reasoning_tokens`, or vLLM
    started with `--reasoning-parser`); non-reasoning models and backends that
    do not surface a per-response count return None for that key.
    """

    NAME = "TokenCounts"
    KEYS = ["Completion", "Reasoning"]

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error:
            values: dict[str, float | None] = {"Completion": None, "Reasoning": None}
        else:
            values = {
                "Completion": response.raw_completion_num_tokens,
                "Reasoning": response.raw_completion_reasoning_num_tokens,
            }

        return [
            MetricResult(
                metric_name=f"{self.NAME}/{key}",
                value=value,
                higher_is_better=False,
                error=response.error,
            )
            for key, value in values.items()
        ]
