from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.loglikelihood.base import BaseLoglikelihoodMetric


class ConfidenceWeightedAccuracy(BaseLoglikelihoodMetric):
    NAME = "Confidence-weighted Accuracy"

    def __init__(
        self,
        *,
        assume_len_normalised: bool = False,
    ) -> None:
        super().__init__(assume_len_normalised=assume_len_normalised)

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

        best_key = max(loglikelihoods, key=loglikelihoods.get)  # type: ignore[arg-type]
        best_key_norm = self._normalise_text(best_key)
        p_c = probs.get(best_key, 0.0)

        accuracy = p_c if best_key_norm in ground_truths else 0.0

        return [MetricResult(metric_name=self.NAME, value=accuracy, higher_is_better=True, error=response.error)]
