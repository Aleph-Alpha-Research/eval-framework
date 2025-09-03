from eval_framework.exceptions import LogicError
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.metrics.llm.graders.language import AVAILABLE_LANGUAGES
from eval_framework.shared.types import Completion


class LanguageChecker(BaseMetric[Completion]):
    NAME = "Language Check"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        if response.ground_truth is None:
            raise LogicError("Language detection needs ground_truth.")
        if response.ground_truth not in AVAILABLE_LANGUAGES:
            raise LogicError("Checking for unknown or unavailable language.")

        completion_language = response.get_completion_language()
        target_language = response.ground_truth
        value = float(completion_language == target_language)
        return [MetricResult(metric_name=self.NAME, value=value, higher_is_better=True, error=response.error)]


class GermanCompletionChecker(BaseMetric[Completion]):
    NAME = "German Completion Check"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        raw_completion_language = response.get_raw_completion_language()
        value = float(raw_completion_language == "de")
        return [MetricResult(metric_name=self.NAME, value=value, higher_is_better=True, error=response.error)]


class LanguageConsistencyChecker(BaseMetric[Completion]):
    NAME = "Language Consistency"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        completion_language = response.get_completion_language()
        target_language = response.get_instruction_language()
        if completion_language == target_language == "":
            return []  # No language information could be determined
        else:
            value = float(completion_language == target_language)
            return [MetricResult(metric_name=self.NAME, value=value, higher_is_better=True, error=response.error)]


class LanguageRawConsistencyChecker(BaseMetric[Completion]):
    NAME = "Language Consistency Raw"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        raw_completion_language = response.get_raw_completion_language()
        target_language = response.get_instruction_language()

        if raw_completion_language == target_language == "":
            return []  # No language information could be determined
        else:
            value = float(raw_completion_language == target_language)
            return [
                MetricResult(
                    metric_name=self.NAME,
                    value=value,
                    higher_is_better=True,
                    error=response.error,
                )
            ]
