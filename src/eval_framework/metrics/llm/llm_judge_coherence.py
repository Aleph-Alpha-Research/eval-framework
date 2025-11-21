from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm.base import BaseLLMJudgeMetric
from eval_framework.metrics.llm.graders.coherence_grader import CoherenceGrader
from eval_framework.metrics.llm.graders.language import Language
from eval_framework.shared.types import Completion


class LLMJudgeCoherence(BaseLLMJudgeMetric):
    NAME = "Coherence"
    KEYS = [
        "coherence_score",
    ]

    def __init__(self, llm_judge: BaseLLM):
        super().__init__(llm_judge)
        self._grader = CoherenceGrader(llm_judge)

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            for key in self.KEYS:
                return [
                    MetricResult(
                        metric_name=f"{self.NAME} - {key}", value=None, higher_is_better=True, error=response.error
                    )
                ]

        language = Language(response.get_instruction_language())

        grading = self._grader.grade(
            instruction=response.system_user_instruction,
            completion=response.sanitized_completion,
            language=language,
        )

        result = MetricResult(
            metric_name=f"{self.NAME}/coherence_score",
            value=grading.coherence_score,
            higher_is_better=True,
            llm_judge_prompt=grading.judge_prompt,
            llm_judge_response=grading.judge_response,
            error=response.error,
        )
        return [result]
