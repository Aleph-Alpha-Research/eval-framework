import numpy as np

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Loglikelihood


class ProbabilityMass(BaseMetric[Loglikelihood]):
    NAME = "Probability Mass"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        assert isinstance(response.ground_truth, str)
        # https://docs.python.org/3.10/library/stdtypes.html?highlight=dictview#dictionary-view-objects
        possible_completions = list(response.loglikelihoods.keys())
        ground_truth_index = possible_completions.index(response.ground_truth)
        split_idx = ground_truth_index + 1

        log_probs = list(response.loglikelihoods.values())
        probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
        prob_mass = np.sum(probs[:split_idx])

        return [
            MetricResult(metric_name=self.NAME, value=float(prob_mass), higher_is_better=True, error=response.error)
        ]


class ProbabilityMassNorm(BaseMetric[Loglikelihood]):
    NAME = "Probability Mass Normalized"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        assert isinstance(response.ground_truth, str)
        # len normalized

        output_len_normalized = {}
        for k, v in response.loglikelihoods.items():
            completion_length = len(k)

            if completion_length != 0:
                output_len_normalized[k] = v / completion_length
            else:
                output_len_normalized[k] = v

        possible_completions = list(response.loglikelihoods.keys())
        ground_truth_index = possible_completions.index(response.ground_truth)
        split_idx = ground_truth_index + 1

        log_probs = list(output_len_normalized.values())
        probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
        prob_mass_norm = np.sum(probs[:split_idx])

        return [MetricResult(metric_name=self.NAME, value=prob_mass_norm, higher_is_better=True, error=response.error)]
