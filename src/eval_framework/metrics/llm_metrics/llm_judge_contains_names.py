from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm_metrics.base import BaseLLMJudgeMetric
from eval_framework.metrics.llm_metrics.graders.contains_names_grader import ContainsNamesGrader
from eval_framework.metrics.llm_metrics.graders.language import Language
from eval_framework.shared.types import Completion


class LLMJudgeAvoidsNames(BaseLLMJudgeMetric):
    NAME = "Avoids Names"

    def __init__(self, llm_judge: BaseLLM):
        super().__init__(llm_judge)
        self._grader = ContainsNamesGrader(llm_judge)

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        language = Language(response.get_instruction_language())

        grading = self._grader.grade(
            completion=response.sanitized_completion,
            language=language,
        )

        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(not grading.contains_names) if grading.contains_names is not None else None,
                higher_is_better=True,
                llm_judge_prompt=grading.judge_prompt,
                llm_judge_response=grading.judge_response,
                error=response.error,
            )
        ]
