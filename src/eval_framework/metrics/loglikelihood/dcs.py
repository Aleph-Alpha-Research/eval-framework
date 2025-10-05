import numpy as np

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Loglikelihood


class DistributionalCorrectnessScore(BaseMetric[Loglikelihood]):
    NAME = "Distributional Correctness Score"

    LC: float = 1.0 # Default reward weight for correct answers
    LW: float = 1.0 # Default penalty weight for wrong answers

    def _normalise_text(self, text: str) -> str:
        return text.strip().lower()

    def __init__(
        self,
        *,
        lc: float | None = None,
        lw: float | None = None,
        assume_len_normalised: bool = False,
    ) -> None:
        self._lc = float(lc) if lc is not None else float(self.LC)
        self._lw = float(lw) if lw is not None else float(self.LW)
        if not (self._lc >= 0 and self._lw >= 0 and self._lc >= self._lw):
            raise ValueError(
                f"Invalid DCS loadings: lc={self._lc}, lw={self._lw}. Require lc>=0, lw>=0, and lc>=lw."
            )
        self._assume_len_normalised = assume_len_normalised

    def _length_normalise_loglikelihoods(self, loglikelihoods: dict) -> dict:
        output = {}
        for k, v in loglikelihoods.items():
            length = len(k)
            output[k] = v / length if length > 0 else v
        return output

    def _softmax(self, log_probs: dict) -> dict:
        vals = list(log_probs.values())
        m = max(vals)
        exp_vals = [math.exp(x - m) for x in vals]
        total = sum(exp_vals)
        return {k: ev / total for k, ev in zip(log_probs.keys(), exp_vals)}

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        loglikelihoods = response.loglikelihoods if self._assume_len_normalised else self._length_normalise_loglikelihoods(response.loglikelihoods)
        probs = self._softmax(loglikelihoods)

        if not loglikelihoods or not probs:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=None)]

        ground_truths = set(
            self._normalise_text(gt) for gt in (response.ground_truth if isinstance(response.ground_truth, list) else [response.ground_truth])
        )

        idk_key = self._normalise_text(list(response.loglikelihoods.keys())[-1])

        p_c = sum(p for k, p in probs.items() if self._normalise_text(k) in ground_truths)
        p_idk = probs.get(idk_key, 0.0)
        p_w = sum(p for k, p in probs.items() if self._normalise_text(k) not in ground_truths and self._normalise_text(k) != idk_key)

        dcs = (self._lc * p_c - self._lw * p_w) * (1.0 - p_idk)

        return [MetricResult(metric_name=self.NAME, value=float(dcs), higher_is_better=True, error=response.error)]
