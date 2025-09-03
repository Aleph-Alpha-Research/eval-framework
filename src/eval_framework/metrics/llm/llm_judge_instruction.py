from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm_metrics.base import BaseLLMJudgeMetric
from eval_framework.metrics.llm_metrics.graders.instruction_grader import InstructionGrader
from eval_framework.metrics.llm_metrics.graders.language import Language
from eval_framework.shared.types import Completion


class LLMJudgeInstruction(BaseLLMJudgeMetric):
    NAME = "Instruction Following"
    KEYS = [
        "quality",
        "is_following_instruction",
        "has_correct_grammar_and_spelling",
        "is_context_consistent",
        "is_not_repeating",
        "is_trustworthy",
        "is_safe",
    ]

    def __init__(self, llm_judge: BaseLLM):
        super().__init__(llm_judge)
        self._grader = InstructionGrader(llm_judge)

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

        results = []
        for key in self.KEYS:
            if key == "quality":
                # [0, 1] normalization required for visualizer
                value = (float(getattr(grading, key)) - 1) / 4 if getattr(grading, key) is not None else None
            else:
                value = float(getattr(grading, key)) if getattr(grading, key) is not None else None
            result = MetricResult(
                metric_name=f"{self.NAME}/{key}",
                value=value,
                higher_is_better=True,
                llm_judge_prompt=grading.judge_prompt,
                llm_judge_response=grading.judge_response,
                error=response.error,
            )
            results.append(result)
        return results
