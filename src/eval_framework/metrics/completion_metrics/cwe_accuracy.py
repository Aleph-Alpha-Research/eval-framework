import re

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion, Error


class CWEAccuracy(BaseMetric[Completion]):
    """Metric for Common Word Extraction tasks"""

    NAME = "CWEAccuracy"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        ground_truths = [gt for gt in response.ground_truth_list if gt is not None]
        if not ground_truths:
            return [MetricResult(metric_name=self.NAME, value=0.0, higher_is_better=True, error=response.error)]

        try:
            # Get model's answer
            model_answer = response.completion

            # Check if all words in the correct answer are present in the model's answer
            is_correct = self._is_answer_correct(ground_truths, model_answer)

            return [
                MetricResult(
                    metric_name=self.NAME, value=1.0 if is_correct else 0.0, higher_is_better=True, error=response.error
                )
            ]
        except Exception as e:
            error = Error(error_class=e.__class__.__name__, message=str(e), traceback="")
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=error)]

    def _is_answer_correct(self, correct_answer: list[str], model_answer: str) -> bool:
        """Check if all words in correct_answer are present in model_answer as whole words"""
        model_answer = model_answer.strip().lower()
        correct_answer = [correct.strip().lower() for correct in correct_answer]

        # For each word in the correct answer, check if it exists as a whole word in the model answer
        for word in correct_answer:
            # Create a regex pattern that matches the word as a whole word
            # \b represents a word boundary
            pattern = r"\b" + re.escape(word) + r"\b"
            if not re.search(pattern, model_answer):
                return False

        return True
