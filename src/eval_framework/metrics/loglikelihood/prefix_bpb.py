"""Deprecated wrapper for prefix BPB. Use ``BitsPerByteLoglikelihood``."""

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.metrics.loglikelihood.bpb_common import (
    ALIAS_RULE,
    K0,
    K_ENTRY,
    K_RATE,
    compute_prefix_bpb_results,
    select_gold,
)
from eval_framework.shared.types import Error, Loglikelihood

__all__ = [
    "ALIAS_RULE",
    "K0",
    "K_ENTRY",
    "K_RATE",
    "PrefixBitsPerByte",
    "compute_prefix_bpb_results",
]


class PrefixBitsPerByte(BaseMetric[Loglikelihood]):
    """Use ``BitsPerByteLoglikelihood`` instead."""

    NAME = "PrefixBPB"
    _SIDECARS = ("PrefixBPB_rate", "PrefixBPB_entryCost", "PrefixBPB_contentBytes")

    def _all_error(self, message: str, response: Loglikelihood) -> list[MetricResult]:
        err = response.error or Error(error_class="ValueError", message=message, traceback="")
        names = (self.NAME, *self._SIDECARS)
        return [MetricResult(metric_name=n, value=None, higher_is_better=False, error=err) for n in names]

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error:
            return self._all_error("upstream error", response)

        gold = select_gold(response)
        if gold is None:
            return self._all_error("No ground-truth answer found in loglikelihoods", response)

        results = compute_prefix_bpb_results(response, gold)
        if not results:
            return self._all_error("No per-token logprobs available for the gold (backend did not emit them)", response)
        return results
