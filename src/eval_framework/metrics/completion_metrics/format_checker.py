import json
import re

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class CheckJsonFormat(BaseMetric[Completion]):
    NAME = "JSON Format"

    def _preprocess(self, completion: str) -> str:
        completion = completion.strip()
        for prefix in ["```json", "```Json", "```JSON", "```"]:
            completion = completion.removeprefix(prefix)
        completion = completion.removesuffix("```")
        completion = completion.strip()
        return completion

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        json_text = self._preprocess(response.completion)

        try:
            json.loads(json_text)
            is_valid_json = True
        except ValueError as _:
            is_valid_json = False

        return [
            MetricResult(metric_name=self.NAME, value=float(is_valid_json), higher_is_better=True, error=response.error)
        ]


class CheckPostScriptFormat(BaseMetric[Completion]):
    """
    This metric is honestly not that great
    In the original IFEval implementation it just checks whether the
    text contains the string (P.)P.S. or variants thereof such as p. s.
    It doesn't check for parsing
    """

    NAME = "Postscript Format"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        postscript_pattern = r"\s*(P\.S\.|P\.P\.S\.)"
        postscript = re.findall(postscript_pattern, response.completion, flags=re.MULTILINE)
        return [
            MetricResult(
                metric_name=self.NAME, value=1.0 if postscript else 0.0, higher_is_better=True, error=response.error
            )
        ]
