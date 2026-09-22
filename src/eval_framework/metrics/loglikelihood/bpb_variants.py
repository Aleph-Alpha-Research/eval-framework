"""Extended bits-per-byte for loglikelihood responses.

Emits the standard ``BitsPerByte`` ratio, corpus-aggregation fields, and prefix
BPB when per-token logprobs exist.
"""

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.metrics.loglikelihood.bpb_variants_common import compute_all_bpb_results
from eval_framework.shared.types import Loglikelihood


class BitsPerByteVariantsLoglikelihood(BaseMetric[Loglikelihood]):
    """Standard BPB plus prefix, prior, and corpus companion fields."""

    NAME = "BitsPerByteVariants"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        return compute_all_bpb_results(response)
