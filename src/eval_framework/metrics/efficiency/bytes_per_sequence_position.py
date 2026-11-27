from eval_framework.metrics.base import BaseMetric, JudgeProvider, MetricResult, no_judge
from eval_framework.shared.types import Completion, Loglikelihood


class BytesLoglikelihood(BaseMetric[Loglikelihood]):
    NAME = "Bytes"

    def calculate(self, response: Loglikelihood, *, judge: JudgeProvider = no_judge) -> list[MetricResult]:
        if response.concat_compression is None:
            value = None
        else:
            value = response.concat_compression.num_bytes

        return [MetricResult(metric_name=self.NAME, value=value, higher_is_better=True)]


class SequencePositionsLoglikelihood(BaseMetric[Loglikelihood]):
    NAME = "SequencePositions"

    def calculate(self, response: Loglikelihood, *, judge: JudgeProvider = no_judge) -> list[MetricResult]:
        if response.concat_compression is None:
            value = None
        else:
            value = response.concat_compression.num_tokens
        return [MetricResult(metric_name=self.NAME, value=value, higher_is_better=True)]


class BytesCompletion(BaseMetric[Completion]):
    NAME = "Bytes"

    def calculate(self, response: Completion, *, judge: JudgeProvider = no_judge) -> list[MetricResult]:
        if response.concat_compression is None:
            value = None
        else:
            value = response.concat_compression.num_bytes

        return [MetricResult(metric_name=self.NAME, value=value, higher_is_better=True)]


class SequencePositionsCompletion(BaseMetric[Completion]):
    NAME = "SequencePositions"

    def calculate(self, response: Completion, *, judge: JudgeProvider = no_judge) -> list[MetricResult]:
        if response.concat_compression is None:
            value = None
        else:
            value = response.concat_compression.num_tokens
        return [MetricResult(metric_name=self.NAME, value=value, higher_is_better=True)]
