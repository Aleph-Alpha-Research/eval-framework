import sacrebleu

from eval_framework.exceptions import LogicError
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class TER(BaseMetric[Completion]):
    """Translation Error Rate is an error metric for machine translation that
    measures the number of edits required to change a system output into one
    of the references
    Source: http://www.cs.umd.edu/~snover/tercom/
    Paper: http://mt-archive.info/AMTA-2006-Snover.pdf
    """

    NAME = "TER"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=False, error=response.error)]

        scores = []
        for ground_truth in response.ground_truth_list:
            if ground_truth == "" or ground_truth is None:
                raise LogicError("When calculating TER we need a ground truth.")

            sacre_formatted_completion = [response.completion]
            sacre_formatted_ground_truth = [[ground_truth]]
            ter_score = sacrebleu.corpus_ter(sacre_formatted_completion, sacre_formatted_ground_truth).score
            scores.append(ter_score)

        return [
            MetricResult(metric_name=self.NAME, value=float(min(scores)), higher_is_better=False, error=response.error)
        ]


class LINEWISE_TER(BaseMetric[Completion]):
    """Minimum Line-level TER (Translation Edit Rate) score."""

    NAME = "Linewise TER"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=False, error=response.error)]

        scores = []
        for ground_truth in response.ground_truth_list:
            for sentence in response.completion.split("\n"):
                if sentence == "":
                    continue

                if ground_truth == "" or ground_truth is None:
                    raise LogicError("When calculating TER we need a ground truth.")

                sacre_formatted_completion = [sentence]
                sacre_formatted_ground_truth = [[ground_truth]]
                ter_score = sacrebleu.corpus_ter(sacre_formatted_completion, sacre_formatted_ground_truth).score
                scores.append(ter_score)

        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(min(scores, default=100)),
                higher_is_better=False,
                error=response.error,
            )
        ]
