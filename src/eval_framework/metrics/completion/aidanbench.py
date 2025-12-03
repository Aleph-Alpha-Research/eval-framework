import logging

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion

logger = logging.getLogger(__name__)


class AidanBenchMetric(BaseMetric[Completion]):
    NAME = "AidanBench"

    def calculate(self, response: Completion) -> list[MetricResult]:
        # subtract 2 to not count 1) initial instruction and 2) the latest model response, which caused the stop
        # i.e. was not (unique && coherent)
        num_unique_responses = len(response.messages) - 2 if response.messages is not None else 0
        if num_unique_responses < 0:
            logger.warning(
                "Number of unique responses calculated as negative, setting to 0."
                "Likely something went wrong during answer generation."
            )
            num_unique_responses = 0
        return [
            MetricResult(
                metric_name=f"{self.NAME}/num_responses",
                value=num_unique_responses,
                higher_is_better=True,
            )
        ]
