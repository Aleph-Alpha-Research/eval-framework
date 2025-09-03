from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import (
    MetricResult,
)
from eval_framework.metrics.llm_metrics.base import BaseLLMJudgeMetric
from eval_framework.metrics.llm_metrics.graders.format_correctness_grader import FormatCorrectnessGrader
from eval_framework.metrics.llm_metrics.graders.language import Language
from eval_framework.shared.types import BaseMetricContext, Completion, LanguageMetricContext, extract_context_metric


class LLMJudgeFormatCorrectnessContext(BaseMetricContext):
    language: str


class LLMJudgeFormatCorrectness(BaseLLMJudgeMetric):
    NAME = "Format Correctness"

    def __init__(self, llm_judge: BaseLLM):
        super().__init__(llm_judge)
        self._grader = FormatCorrectnessGrader(llm_judge)

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        context = extract_context_metric(response, LanguageMetricContext)

        grading = self._grader.grade(
            instruction=response.system_user_instruction,
            completion=response.sanitized_completion,
            language=Language(context.language),
        )

        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(grading.format_correctness) if grading.format_correctness is not None else None,
                higher_is_better=True,
                llm_judge_prompt=grading.judge_prompt,
                llm_judge_response=grading.judge_response,
                error=response.error,
            )
        ]
