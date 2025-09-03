from eval_framework.exceptions import LogicError
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class ROUGE_L(BaseMetric[Completion]):
    """ROUGE-L"""

    NAME = "ROUGE-L"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        if response.completion == "":
            return [MetricResult(metric_name=self.NAME, value=0.0, higher_is_better=True, error=response.error)]
        if None in response.ground_truth_list:
            raise LogicError("When calculating ROUGE-L ground_truth cannot be None.")

        # ROUGE-L is essentially an F1 score, but it’s a specific F1 score based on
        # the Longest Common Subsequence (LCS) between a candidate summary and a reference summary.
        rouge = max([_calculate_rouge_l(response.completion, gt) for gt in response.ground_truth_list])  # type: ignore[arg-type]
        return [MetricResult(metric_name=self.NAME, value=float(rouge), higher_is_better=True, error=response.error)]


def _longest_common_subsequence_length(candidate_tokens: list[str], reference_tokens: list[str]) -> int:
    candidate_len, reference_len = len(candidate_tokens), len(reference_tokens)
    lcs_matrix = [[0] * (reference_len + 1) for _ in range(candidate_len + 1)]

    for i in range(candidate_len + 1):
        for j in range(reference_len + 1):
            if i == 0 or j == 0:
                lcs_matrix[i][j] = 0
            elif candidate_tokens[i - 1] == reference_tokens[j - 1]:
                lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i - 1][j], lcs_matrix[i][j - 1])

    return lcs_matrix[candidate_len][reference_len]


def _calculate_rouge_l(completion: str, ground_truth: str) -> float:
    lcs_length = _longest_common_subsequence_length(completion.split(), ground_truth.split())
    if lcs_length == 0:
        return 0.0
    precision = lcs_length / len(completion.split())
    recall = lcs_length / len(ground_truth.split())
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score
