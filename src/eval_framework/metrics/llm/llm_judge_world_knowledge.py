from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm.base import BaseLLMJudgeMetric
from eval_framework.metrics.llm.graders.language import Language
from eval_framework.metrics.llm.graders.summary_world_knowledge_grader import SummarizationWorldKnowledgeGrader
from eval_framework.shared.types import Completion


class LLMJudgeWorldKnowledge(BaseLLMJudgeMetric):
    NAME = "World Knowledge"

    def __init__(self, llm_judge: BaseLLM):
        super().__init__(llm_judge)
        self._grader = SummarizationWorldKnowledgeGrader(llm_judge)

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=False, error=response.error)]

        language = Language(response.get_instruction_language())

        grading = self._grader.grade(
            reference_input=response.user_instruction,
            completion=response.sanitized_completion,
            language=language,
        )

        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(grading.contains_world_knowledge) if grading.contains_world_knowledge is not None else None,
                higher_is_better=False,
                llm_judge_prompt=grading.judge_prompt,
                llm_judge_response=grading.judge_response,
                error=response.error,
            )
        ]
