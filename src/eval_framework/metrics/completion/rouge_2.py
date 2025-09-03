from eval_framework.exceptions import LogicError
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.metrics.completion.f1 import calculate_f1
from eval_framework.shared.types import Completion


class ROUGE_2(BaseMetric[Completion]):
    """ROUGE-2"""

    NAME = "ROUGE-2"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        if response.completion == "":
            return [MetricResult(metric_name=self.NAME, value=0.0, higher_is_better=True, error=response.error)]
        if None in response.ground_truth_list:
            raise LogicError("When calculating ROUGE-2 ground_truth cannot be None.")

        # ROUGE-2 captures word sequence similarity by focusing on bigrams,
        # which makes it sensitive to the order and co-occurrence of words to some extent.
        rouge = max([_calculate_rouge_2(response.completion, gt) for gt in response.ground_truth_list])  # type: ignore[arg-type]
        return [MetricResult(metric_name=self.NAME, value=float(rouge), higher_is_better=True, error=response.error)]


def _generate_bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    """Generate bigrams from a list of tokens."""
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def _calculate_rouge_2(completion: str, ground_truth: str) -> float:
    """
    Calculate ROUGE-2 precision, recall, and F1 score between candidate and reference texts.
    """

    # Tokenize the candidate and reference summaries
    candidate_tokens = completion.split()
    reference_tokens = ground_truth.split()

    # Generate bigrams for candidate and reference
    candidate_bigrams = _generate_bigrams(candidate_tokens)
    reference_bigrams = _generate_bigrams(reference_tokens)

    return calculate_f1(reference_bigrams, candidate_bigrams)
