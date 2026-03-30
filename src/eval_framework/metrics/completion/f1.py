import re
import string
from collections import Counter
from typing import Any

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class F1(BaseMetric[Completion]):
    """
    Token-overlap F1 metric.

    Default behavior is backward-compatible with the previous implementation:
    lowercase + whitespace tokenization.
    """

    NAME = "F1"

    def normalize(self, text: str) -> str:
        """Normalizes text to use lower case."""
        return text.lower()

    def tokenize(self, text: str) -> list[str]:
        """Tokenizes text into a list of tokens using whitespace as the delimiter."""
        return self.normalize(text).split()

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        ground_truths = [gt for gt in response.ground_truth_list if gt is not None]
        if not ground_truths:
            return [MetricResult(metric_name=self.NAME, value=0.0, higher_is_better=True, error=response.error)]

        prediction_tokens = self.tokenize(response.completion)
        ground_truths_tokens = [self.tokenize(gt) for gt in ground_truths]
        f1_scores = [calculate_f1(gt_tokens, prediction_tokens) for gt_tokens in ground_truths_tokens]
        max_f1 = max(f1_scores)

        return [MetricResult(metric_name=self.NAME, value=max_f1, higher_is_better=True, error=response.error)]


class F1SquadNormalized(F1):
    """
    SQuAD-style normalized F1:
    - lowercase
    - remove punctuation
    - remove articles (a, an, the)
    - collapse extra whitespace
    """

    NAME = "F1 SQuAD Normalized"
    _ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
    _PUNCTUATION = set(string.punctuation)

    def normalize(self, text: str) -> str:
        text = text.lower()
        text = "".join(ch for ch in text if ch not in self._PUNCTUATION)
        text = self._ARTICLES_RE.sub(" ", text)
        text = " ".join(text.split())
        return text


def calculate_f1(ref_tokens: list[Any], hyp_tokens: list[Any]) -> float:
    """Calculate F1 score between two texts based on token overlap."""
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0

    common = Counter(ref_tokens) & Counter(hyp_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(hyp_tokens)
    recall = num_same / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)
