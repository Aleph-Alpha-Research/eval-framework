from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class FinishReason(BaseMetric[Completion]):
    """Why the backend ended the generation, as one 0/1 indicator per reason.

    Averaged over a benchmark, each key becomes a rate: `FinishReason/Repetition`
    is the share of completions vLLM aborted through its `repetition_detection`
    sampling parameter (looping output), `FinishReason/Length` the share that hit
    `max_tokens`, and `FinishReason/Stop` the share that ended on their own (EOS
    or a stop sequence). Reasons outside these keys (e.g. `tool_calls`) count as
    0 for all of them.

    Reads the `finish_reason` backends already attach to the response. All values
    are None when the backend did not report one (e.g. HuggingFace) or when the
    sample errored, so such samples do not dilute the rates.
    """

    NAME = "FinishReason"
    KEYS = ["Stop", "Length", "Repetition"]
    _REASONS = {"Stop": "stop", "Length": "length", "Repetition": "repetition"}

    def calculate(self, response: Completion) -> list[MetricResult]:
        finish_reason = response.raw_completion_finish_reason
        if response.error or finish_reason is None:
            values: dict[str, float | None] = {key: None for key in self.KEYS}
        else:
            values = {key: float(finish_reason.lower() == reason) for key, reason in self._REASONS.items()}

        return [
            MetricResult(
                metric_name=f"{self.NAME}/{key}",
                value=value,
                higher_is_better=key == "Stop",
                error=response.error,
            )
            for key, value in values.items()
        ]
