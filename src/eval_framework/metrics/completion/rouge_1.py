from eval_framework.exceptions import LogicError
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.metrics.completion_metrics.f1 import calculate_f1
from eval_framework.shared.types import Completion


class ROUGE_1(BaseMetric[Completion]):
    """ROUGE-1"""

    NAME = "ROUGE-1"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        if response.completion == "":
            return [MetricResult(metric_name=self.NAME, value=0.0, higher_is_better=True, error=response.error)]
        if None in response.ground_truth_list:
            raise LogicError("When calculating ROUGE-1 ground_truth cannot be None.")

        # ROUGE-1 captures word sequence similarity by focusing on unigrams
        rouge = max([_calculate_rouge_1(response.completion, gt) for gt in response.ground_truth_list])  # type: ignore[arg-type]
        return [MetricResult(metric_name=self.NAME, value=float(rouge), higher_is_better=True, error=response.error)]


def _calculate_rouge_1(candidate: str, reference: str) -> float:
    """
    Calculate ROUGE-1 precision, recall, and F1 score between candidate and reference texts.
    """

    # Tokenize the candidate and reference summaries
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()

    return calculate_f1(reference_tokens, candidate_tokens)
