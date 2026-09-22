"""Deprecated ``BitsPerByteVariantsLoglikelihood`` wrapper."""

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.metrics.loglikelihood.bpb_variants_common import (
    ALIAS_RULE,
    compute_all_bpb_results,
    compute_standard_bpb_results,
    list_ground_truth_candidates,
    select_ground_truth,
    standard_bpb_error,
)
from eval_framework.shared.types import Loglikelihood

__all__ = [
    "ALIAS_RULE",
    "InstrumentedBitsPerByte",
    "compute_standard_bpb_results",
    "list_ground_truth_candidates",
    "select_ground_truth",
    "standard_bpb_error",
]


class InstrumentedBitsPerByte(BaseMetric[Loglikelihood]):
    """Use ``BitsPerByteVariantsLoglikelihood`` instead."""

    NAME = "BitsPerByteVariants"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        results = compute_all_bpb_results(response)
        return [r for r in results if r.metric_name.startswith("BitsPerByte")]
