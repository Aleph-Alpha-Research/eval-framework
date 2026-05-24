from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm.base import BaseLLMJudgeMetric, safe_metric_calculation
from eval_framework.metrics.llm.graders.language import Language
from eval_framework.metrics.llm.graders.refusal_grader import RefusalGrader
from eval_framework.shared.types import Completion


class LLMJudgeRefusal(BaseLLMJudgeMetric):
    NAME = "refusal_classifier"

    def __init__(self, llm_judge: BaseLLM):
        super().__init__(llm_judge)
        self._grader = RefusalGrader(llm_judge)

    @safe_metric_calculation
    def calculate(self, response: Completion) -> list[MetricResult]:
        language = response.get_completion_language() or response.get_instruction_language() or "en"
        grading = self._grader.grade(
            completion=response.sanitized_completion,
            language=Language(language),
        )

        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(grading.is_refusal) if grading.is_refusal is not None else None,
                higher_is_better=True,
                llm_judge_prompt=grading.judge_prompt,
                llm_judge_response=grading.judge_response,
            )
        ]
