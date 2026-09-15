"""Bits-per-byte for loglikelihood responses.

Emits ``BitsPerByte``, corpus-aggregation fields, and prefix BPB when per-token
logprobs exist.
"""

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.metrics.loglikelihood.bpb_common import compute_all_bpb_results
from eval_framework.shared.types import Loglikelihood


class BitsPerByteLoglikelihood(BaseMetric[Loglikelihood]):
    """Negative log-likelihood of the answer divided by its UTF-8 byte length."""

    NAME = "BitsPerByte"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        return compute_all_bpb_results(response)
