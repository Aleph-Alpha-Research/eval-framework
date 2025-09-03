import sacrebleu

from eval_framework.exceptions import LogicError
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class CHRF(BaseMetric[Completion]):
    """chrF++ is a tool for automatic evaluation of machine translation output
    based on character n-gram precision and recall enhanced with word n-grams.
    Source: https://github.com/m-popovic/chrF
    Paper: https://www.aclweb.org/anthology/W15-3049.pdf
    """

    NAME = "chrF"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        scores = []
        for ground_truth in response.ground_truth_list:
            if ground_truth == "" or ground_truth is None:
                raise LogicError("When calculating chrF we need a ground truth.")

            sacre_formatted_completion = [response.completion]
            sacre_formatted_ground_truth = [[ground_truth]]
            scores.append(sacrebleu.corpus_chrf(sacre_formatted_completion, sacre_formatted_ground_truth).score)

        return [
            MetricResult(metric_name=self.NAME, value=float(max(scores)), higher_is_better=True, error=response.error)
        ]


class LINEWISE_CHRF(BaseMetric[Completion]):
    """Maximum Line-level chrF++ (Character n-gram F-score) score.
    Paper: https://aclanthology.org/W15-3049/"""

    NAME = "Linewise chrF"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        scores = []
        for ground_truth in response.ground_truth_list:
            for sentence in response.completion.split("\n"):
                if sentence == "":
                    continue

                if ground_truth == "" or ground_truth is None:
                    raise LogicError("When calculating chrF we need a ground truth.")

                sacre_formatted_completion = [sentence]
                sacre_formatted_ground_truth = [[ground_truth]]
                scores.append(sacrebleu.corpus_chrf(sacre_formatted_completion, sacre_formatted_ground_truth).score)

        return [
            MetricResult(
                metric_name=self.NAME, value=float(max(scores, default=0)), higher_is_better=True, error=response.error
            )
        ]
