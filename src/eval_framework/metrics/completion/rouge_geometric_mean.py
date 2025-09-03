from eval_framework.exceptions import LogicError
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.metrics.completion.rouge_1 import ROUGE_1
from eval_framework.metrics.completion.rouge_2 import ROUGE_2
from eval_framework.metrics.completion.rouge_l import ROUGE_L
from eval_framework.shared.types import Completion


class ROUGE_GEOMETRIC_MEAN(BaseMetric[Completion]):
    """ROUGE Geometric Mean"""

    NAME = "ROUGE-Geometric-Mean"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]
        if response.completion == "":
            return [MetricResult(metric_name=self.NAME, value=0.0, higher_is_better=True, error=response.error)]
        if any(gt is None for gt in response.ground_truth_list):
            raise LogicError("When calculating ROUGE Geometric Mean ground_truth cannot be None.")

        # Calculate ROUGE-1, ROUGE-2, and ROUGE-L
        rouge_1 = ROUGE_1().calculate(response)[0].value
        rouge_2 = ROUGE_2().calculate(response)[0].value
        rouge_l = ROUGE_L().calculate(response)[0].value

        # Calculate the geometric mean of ROUGE-1, ROUGE-2, and ROUGE-L
        if rouge_1 is None or rouge_2 is None or rouge_l is None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        geometric_mean = (rouge_1 * rouge_2 * rouge_l) ** (1 / 3)
        return [
            MetricResult(
                metric_name=self.NAME, value=float(geometric_mean), higher_is_better=True, error=response.error
            )
        ]
