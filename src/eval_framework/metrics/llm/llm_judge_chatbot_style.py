from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm.base import BaseLLMJudgeMetric, safe_metric_calculation
from eval_framework.metrics.llm.graders.chatbot_style_grader import ChatbotStyleGrader
from eval_framework.metrics.llm.graders.language import Language
from eval_framework.shared.types import Completion


class LLMJudgeChatbotStyle(BaseLLMJudgeMetric):
    NAME = "Chatbot Style"

    def __init__(self, llm_judge: BaseLLM):
        super().__init__(llm_judge)
        self._grader = ChatbotStyleGrader(llm_judge)

    @safe_metric_calculation
    def calculate(self, response: Completion) -> list[MetricResult]:
        language = Language(response.get_instruction_language())

        grading = self._grader.grade(
            completion=response.sanitized_completion,
            language=language,
        )

        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(grading.is_chatbot_style) if grading.is_chatbot_style is not None else None,
                higher_is_better=True,
                llm_judge_prompt=grading.judge_prompt,
                llm_judge_response=grading.judge_response,
            )
        ]
