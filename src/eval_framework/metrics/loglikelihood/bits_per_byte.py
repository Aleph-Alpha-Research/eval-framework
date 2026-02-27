import math

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Error, Loglikelihood


class BitsPerByteLoglikelihood(BaseMetric[Loglikelihood]):
    """
    Bits-per-byte metric for loglikelihood responses.

    This follows the Paloma definition: the negative log-likelihood of the
    answer divided by the number of UTF-8 bytes in the answer string.
    """

    NAME = "BitsPerByte"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=False, error=response.error)]

        ground_truth_list = response.ground_truth_list

        # Find a ground-truth string that we have a loglikelihood for.
        log_p_x: float | None = None
        answer_text: str | None = None
        for gt in ground_truth_list:
            if gt is None:
                continue
            if gt in response.loglikelihoods:
                answer_text = gt
                log_p_x = float(response.loglikelihoods[gt])
                break

        if log_p_x is None or answer_text is None:
            return [
                MetricResult(
                    metric_name=self.NAME,
                    value=None,
                    higher_is_better=False,
                    error=response.error
                    or Error(
                        error_class="ValueError",
                        message="No ground-truth answer found in loglikelihoods",
                    ),
                )
            ]

        num_bytes = len(answer_text.encode("utf-8"))
        if num_bytes == 0:
            return [
                MetricResult(
                    metric_name=self.NAME,
                    value=None,
                    higher_is_better=False,
                    error=response.error
                    or Error(
                        error_class="ValueError",
                        message="Ground-truth answer has zero UTF-8 bytes",
                    ),
                )
            ]

        bits_per_byte = -log_p_x / (num_bytes * math.log(2))

        return [
            MetricResult(
                metric_name=self.NAME,
                value=bits_per_byte,
                higher_is_better=False,
                error=response.error,
            )
        ]
