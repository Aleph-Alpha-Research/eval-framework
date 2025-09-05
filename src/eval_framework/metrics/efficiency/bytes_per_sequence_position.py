from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion, Loglikelihood


class BytesLoglikelihood(BaseMetric[Loglikelihood]):
    NAME = "Bytes"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error or response.concat_compression is None:
            value = None
        else:
            value = response.concat_compression.num_bytes

        return [MetricResult(metric_name=self.NAME, value=value, higher_is_better=True, error=response.error)]


class SequencePositionsLoglikelihood(BaseMetric[Loglikelihood]):
    NAME = "SequencePositions"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error or response.concat_compression is None:
            value = None
        else:
            value = response.concat_compression.num_tokens
        return [MetricResult(metric_name=self.NAME, value=value, higher_is_better=True, error=response.error)]


class BytesCompletion(BaseMetric[Completion]):
    NAME = "Bytes"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error or response.concat_compression is None:
            value = None
        else:
            value = response.concat_compression.num_bytes

        return [MetricResult(metric_name=self.NAME, value=value, higher_is_better=True, error=response.error)]


class SequencePositionsCompletion(BaseMetric[Completion]):
    NAME = "SequencePositions"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error or response.concat_compression is None:
            value = None
        else:
            value = response.concat_compression.num_tokens
        return [MetricResult(metric_name=self.NAME, value=value, higher_is_better=True, error=response.error)]
