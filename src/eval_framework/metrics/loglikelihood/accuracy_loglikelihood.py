from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Loglikelihood


class AccuracyLoglikelihood(BaseMetric[Loglikelihood]):
    NAME = "Accuracy Loglikelihood"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        ground_truth_list = response.ground_truth_list
        completion_text = max(response.loglikelihoods, key=response.loglikelihoods.get)  # type: ignore[arg-type]

        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(completion_text in ground_truth_list),
                higher_is_better=True,
                error=response.error,
            )
        ]


class AccuracyNormLoglikelihood(BaseMetric[Loglikelihood]):
    NAME = "Accuracy Normalized Loglikelihood"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        ground_truth_list = response.ground_truth_list

        output_len_normalized = {}
        for k, v in response.loglikelihoods.items():
            completion_length = len(k)

            if completion_length != 0:
                output_len_normalized[k] = v / completion_length
            else:
                output_len_normalized[k] = v

        model_output_len_normalized = max(output_len_normalized, key=output_len_normalized.get)  # type:ignore
        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(model_output_len_normalized in ground_truth_list),
                higher_is_better=True,
                error=response.error,
            )
        ]
