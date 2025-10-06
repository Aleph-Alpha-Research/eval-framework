from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.loglikelihood.base import BaseLoglikelihoodMetric
from eval_framework.shared.types import Loglikelihood


class TernaryScore(BaseLoglikelihoodMetric):
    """Based on Kalai et al. (2025) Why language models hallucinate. arXiv:2509.04664"""

    NAME = "Ternary Score"

    def __init__(
        self,
        *,
        lc: float = 1.0,  # Default reward for correct answers
        lw: float = 1.0,  # Default penalty for wrong answers (note: this will be negated in the score)
        len_normalised: bool = True,
    ) -> None:
        super().__init__(len_normalised=len_normalised)
        self._lc = float(lc)
        self._lw = float(lw)
        if not (self._lc >= 0 and self._lw >= 0):
            raise ValueError(f"Invalid reward and penalty values: lc={self._lc}, lw={self._lw}. Require lc>=0, lw>=0.")

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        if self.len_normalised:
            loglikelihoods = self._length_normalise_loglikelihoods(response.loglikelihoods)
        else:
            loglikelihoods = response.loglikelihoods

        ground_truths = set(
            self._normalise_text(gt)
            for gt in (response.ground_truth if isinstance(response.ground_truth, list) else [response.ground_truth])
        )
        completion_text = max(loglikelihoods, key=loglikelihoods.get)  # type: ignore[arg-type]
        norm_text = self._normalise_text(completion_text)
        idk_key = self._normalise_text(list(response.loglikelihoods.keys())[-1])  # assumes last key is "IDK" option

        if norm_text in ground_truths:
            score = self._lc
        elif norm_text == idk_key:
            score = 0.0
        else:
            score = -self._lw

        return [MetricResult(metric_name=self.NAME, value=score, higher_is_better=True, error=response.error)]
