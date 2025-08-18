import sacrebleu

from eval_framework.exceptions import LogicError
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class BLEU(BaseMetric[Completion]):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/
    """

    NAME = "BLEU"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        scores = []
        for ground_truth in response.ground_truth_list:
            if ground_truth == "" or ground_truth is None:
                raise LogicError("When calculating BLEU we need a ground truth.")

            sacre_formatted_completion = [response.completion]
            sacre_formatted_ground_truth = [[ground_truth]]
            scores.append(sacrebleu.corpus_bleu(sacre_formatted_completion, sacre_formatted_ground_truth).score)

        return [
            MetricResult(metric_name=self.NAME, value=float(max(scores)), higher_is_better=True, error=response.error)
        ]


class LINEWISE_BLEU(BaseMetric[Completion]):
    """Maximum Line-level BLEU score."""

    NAME = "Linewise BLEU"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        scores = []
        for ground_truth in response.ground_truth_list:
            for sentence in response.completion.split("\n"):
                if sentence == "":
                    continue

                if ground_truth == "" or ground_truth is None:
                    raise LogicError("When calculating BLEU we need a ground truth.")

                sacre_formatted_completion = [sentence]
                sacre_formatted_ground_truth = [[ground_truth]]
                scores.append(sacrebleu.corpus_bleu(sacre_formatted_completion, sacre_formatted_ground_truth).score)

        return [
            MetricResult(metric_name=self.NAME, value=float(max(scores)), higher_is_better=True, error=response.error)
        ]


class ResponseToOriginalBLEU(BaseMetric[Completion]):
    NAME = "Response to Original BLEU"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        score = sacrebleu.corpus_bleu([response.completion], [[response.last_user_instruction]]).score
        # scaled to [0, 1] to make aggregation easier
        return [MetricResult(metric_name=self.NAME, value=score / 100, higher_is_better=True, error=response.error)]
