import numpy as np

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Loglikelihood
from eval_framework.utils.helpers import count_bytes


class AccuracyLoglikelihood(BaseMetric[Loglikelihood]):
    NAME = "Accuracy Loglikelihood"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        ground_truth_list = response.ground_truth_list
        completion_text = max(response.loglikelihoods, key=response.loglikelihoods.get)  # type: ignore[arg-type]

        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(completion_text in ground_truth_list),
                higher_is_better=True,
                error=response.error,
            )
        ]


class AccuracyNormLoglikelihood(BaseMetric[Loglikelihood]):
    NAME = "Accuracy Normalized Loglikelihood"

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        ground_truth_list = response.ground_truth_list

        output_len_normalized = {}
        for k, v in response.loglikelihoods.items():
            completion_length = len(k)

            if completion_length != 0:
                output_len_normalized[k] = v / completion_length
            else:
                output_len_normalized[k] = v

        model_output_len_normalized = max(output_len_normalized, key=output_len_normalized.get)  # type:ignore
        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(model_output_len_normalized in ground_truth_list),
                higher_is_better=True,
                error=response.error,
            )
        ]


class AccuracyBayesianLoglikelihood(BaseMetric[Loglikelihood]):
    """Accuracy after adjusting the loglikelihoods for the byte-length bias of the completion.
    See https://arxiv.org/html/2607.12767v1 for more details.
    """

    NAME = "Accuracy Bayesian Loglikelihood"

    def __init__(self) -> None:
        self.length_decay = 0.0

    def prepare(self, responses: list[Loglikelihood]) -> None:
        """
        Estimating the length decay factor.
        See Equation (24) in https://arxiv.org/html/2607.12767v1
        """
        numerator = 0.0
        denominator = 0.0

        for response in responses:
            if response.error is not None:
                continue

            num_candidates = len(response.loglikelihoods)
            if num_candidates <= 1:
                continue

            lengths = np.array([count_bytes(completion) for completion in response.loglikelihoods], dtype=float)
            loglikelihoods = np.array(list(response.loglikelihoods.values()), dtype=float)
            length_differences = lengths - np.mean(lengths)
            loglikelihood_differences = loglikelihoods - np.mean(loglikelihoods)

            local_denominator = num_candidates * np.sum(length_differences**2)
            if local_denominator == 0:
                continue

            numerator += float(num_candidates * np.sum(length_differences * loglikelihood_differences))
            denominator += float(local_denominator)

        self.length_decay = 0.0 if denominator == 0 else numerator / denominator

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        corrected_loglikelihoods = {
            completion: loglikelihood - self.length_decay * count_bytes(completion)
            for completion, loglikelihood in response.loglikelihoods.items()
        }
        completion_text = max(corrected_loglikelihoods, key=corrected_loglikelihoods.get)  # type: ignore[arg-type]

        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(completion_text in response.ground_truth_list),
                higher_is_better=True,
                error=response.error,
            )
        ]


class PartialEvalAccuracy(BaseMetric[Loglikelihood]):
    """An accuracy metric for partial evaluation tasks, e.g. WinograndeCloze.

    Here, for each item, we generate a pair of two samples, one for each option.
    We then calculate the accuracy of the model's completion for each option,
    and then use the accuracy of the correct option to calculate the overall accuracy.

    NOTE: The current implementation relies on the assumption that it comes in pairs of samples,
    which can be identified by having consecutive ids (odd and even). This is how it is implemented
    in the WinograndeCloze tasks, but if other tasks use this metric, it might not be the case and
    require a more general implementation (e.g. storing `item_id` in the `Sample.context`).
    """

    NAME = "Partial Evaluation Accuracy"

    def __init__(self) -> None:
        self._pending: dict[int, tuple[float, bool]] = {}

    def calculate(self, response: Loglikelihood) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        item_id = response.id // 2  # Each item generates 2 samples, get a unique id for the pair
        logprob = next(iter(response.loglikelihoods.values()))
        is_correct = response.ground_truth == "True"

        if item_id not in self._pending:
            # Store the logprob and is_correct for the first sample and wait for the second sample to come in
            self._pending[item_id] = (logprob, is_correct)
            return []
        else:
            # Both samples exist, calculate the accuracy
            other_logprob, other_is_correct = self._pending.pop(item_id)
            # Verify that only one of the samples is correct
            assert other_is_correct != is_correct, "Both samples cannot be correct or incorrect at the same time"

            accuracy = is_correct if logprob > other_logprob else other_is_correct

            return [
                MetricResult(
                    metric_name=self.NAME,
                    value=float(accuracy),
                    higher_is_better=True,
                )
            ]
