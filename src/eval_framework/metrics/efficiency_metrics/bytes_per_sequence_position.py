from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion, Loglikelihood


class BytesLoglikelihood(BaseMetric[Loglikelihood]):
    NAME = "Bytes"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        # sequence positions for prompt as well as loglikelihoods must be present
        positions = [response.prompt_sequence_positions] + list(response.loglikelihoods_sequence_positions.values())
        if response.error is not None or any(v is None for v in positions) or len(positions) <= 1:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        text = response.prompt + "".join([k for k in response.loglikelihoods_sequence_positions.keys()])
        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(len(bytes(text.encode("UTF-8")))),
                higher_is_better=True,
                error=response.error,
            )
        ]


class SequencePositionsLoglikelihood(BaseMetric[Loglikelihood]):
    NAME = "SequencePositions"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        # sequence positions for prompt as well as loglikelihoods must be present
        positions = [response.prompt_sequence_positions] + list(response.loglikelihoods_sequence_positions.values())
        if response.error is not None or any(v is None for v in positions) or len(positions) <= 1:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]
        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(sum(positions)),  # type: ignore
                higher_is_better=True,
                error=response.error,
            )
        ]


class BytesCompletion(BaseMetric[Completion]):
    NAME = "Bytes"

    def calculate(self, response: Completion) -> list[MetricResult]:
        # sequence positions for prompt as well as completion must be present
        positions = [response.prompt_sequence_positions, response.raw_completion_sequence_positions]
        if response.error is not None or any(v is None for v in positions):
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        text = response.prompt + response.raw_completion
        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(len(bytes(text.encode("UTF-8")))),
                higher_is_better=True,
                error=response.error,
            )
        ]


class SequencePositionsCompletion(BaseMetric[Completion]):
    NAME = "SequencePositions"

    def calculate(self, response: Completion) -> list[MetricResult]:
        # sequence positions for prompt as well as completion must be present
        positions = [response.prompt_sequence_positions, response.raw_completion_sequence_positions]
        if response.error is not None or any(v is None for v in positions):
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]
        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(sum(positions)),  # type: ignore
                higher_is_better=True,
                error=response.error,
            )
        ]
