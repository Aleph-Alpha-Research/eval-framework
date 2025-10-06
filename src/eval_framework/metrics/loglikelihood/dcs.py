from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.loglikelihood.base import BaseLoglikelihoodMetric
from eval_framework.shared.types import Loglikelihood


class DistributionalCorrectnessScore(BaseLoglikelihoodMetric):
    """Based on Burns (2025) Measuring Language Model Hallucinations Through Distributional Correctness."""

    NAME = "Distributional Correctness Score"

    def __init__(
        self,
        *,
        lc: float = 1.0,  # Default reward weight for correct answers
        lw: float = 1.0,  # Default penalty weight for wrong answers
        len_normalised: bool = True,
    ) -> None:
        super().__init__(len_normalised=len_normalised)
        self._lc = float(lc)
        self._lw = float(lw)
        if not (self._lc >= 0 and self._lw >= 0 and self._lc >= self._lw):
            raise ValueError(f"Invalid DCS loadings: lc={self._lc}, lw={self._lw}. Require lc>=0, lw>=0, and lc>=lw.")

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        if self.len_normalised:
            loglikelihoods = self._length_normalise_loglikelihoods(response.loglikelihoods)
        else:
            loglikelihoods = response.loglikelihoods
        probs = self._softmax(loglikelihoods)

        ground_truths = set(
            self._normalise_text(gt)
            for gt in (response.ground_truth if isinstance(response.ground_truth, list) else [response.ground_truth])
        )

        idk_key = self._normalise_text(list(response.loglikelihoods.keys())[-1])  # assumes last key is "IDK" option

        p_c = sum(p for k, p in probs.items() if self._normalise_text(k) in ground_truths)
        p_idk = probs.get(idk_key, 0.0)
        p_w = sum(
            p
            for k, p in probs.items()
            if (self._normalise_text(k) not in ground_truths and self._normalise_text(k) != idk_key)
        )

        dcs = (self._lc * p_c - self._lw * p_w) * (1.0 - p_idk)

        return [MetricResult(metric_name=self.NAME, value=float(dcs), higher_is_better=True, error=response.error)]
