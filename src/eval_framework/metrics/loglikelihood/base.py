import math

from eval_framework.metrics.base import BaseMetric
from eval_framework.shared.types import Loglikelihood


class BaseLoglikelihoodMetric(BaseMetric[Loglikelihood]):
    """Base class for metrics that operate on loglikelihood responses."""

    def __init__(
        self,
        *,
        len_normalised: bool = True,
    ) -> None:
        self.len_normalised = len_normalised

    def _normalise_text(self, text: str) -> str:
        return text.strip().lower()

    def _length_normalise_loglikelihoods(self, loglikelihoods: dict) -> dict:
        """Return a dict of length-normalised loglikelihoods."""
        output = {}
        for k, v in loglikelihoods.items():
            length = len(k)
            output[k] = v / length if length > 0 else v
        return output

    def _compute_probabilities(self, loglikelihoods: dict) -> tuple[dict, dict]:
        """Compute probabilities from loglikelihoods, with optional length normalisation."""
        if self.len_normalised:
            loglikelihoods = self._length_normalise_loglikelihoods(loglikelihoods)
        return loglikelihoods, self._softmax(loglikelihoods)

    def _gather_ground_truths(self, response: Loglikelihood) -> set[str]:
        """Extract and normalize ground truth completions from a Loglikelihood response."""
        ground_truths = set(
            self._normalise_text(gt)
            for gt in (response.ground_truth if isinstance(response.ground_truth, list) else [response.ground_truth])
        )
        return ground_truths

    def _softmax(self, log_probs: dict) -> dict:
        """Convert log-likelihoods to probabilities with softmax."""
        vals = list(log_probs.values())
        if not vals:  # no valid entries
            return {}
        m = max(vals)
        exp_vals = [math.exp(x - m) for x in vals]
        total = sum(exp_vals)
        return {k: ev / total for k, ev in zip(log_probs.keys(), exp_vals)}
