from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.loglikelihood.base import BaseLoglikelihoodMetric
from eval_framework.shared.types import Loglikelihood


class ConfidenceWeightedAccuracy(BaseLoglikelihoodMetric):
    NAME = "Confidence-weighted Accuracy"

    def __init__(self, *, len_normalised: bool = True) -> None:
        super().__init__(len_normalised=len_normalised)

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

        best_key = max(loglikelihoods, key=loglikelihoods.get)  # type: ignore[arg-type]
        best_key_norm = self._normalise_text(best_key)
        p_c = probs.get(best_key, 0.0)

        accuracy = p_c if best_key_norm in ground_truths else 0.0

        return [MetricResult(metric_name=self.NAME, value=accuracy, higher_is_better=True, error=response.error)]
