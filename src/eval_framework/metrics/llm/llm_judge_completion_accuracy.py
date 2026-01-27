from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm.base import BaseLLMJudgeMetric, safe_metric_calculation
from eval_framework.metrics.llm.graders.language import Language
from eval_framework.metrics.llm.graders.long_context_grader import LongContextGrader
from eval_framework.shared.types import Completion


class LLMJudgeCompletionAccuracy(BaseLLMJudgeMetric):
    NAME = "Judge Completion Accuracy"

    def __init__(self, llm_judge: BaseLLM):
        super().__init__(llm_judge)
        self._grader = LongContextGrader(llm_judge)

    @safe_metric_calculation
    def calculate(self, response: Completion) -> list[MetricResult]:
        assert isinstance(response.ground_truth, str)

        language = Language(response.get_instruction_language())

        grading = self._grader.grade(
            expected_output=response.ground_truth,
            completion=response.sanitized_completion,
            language=language,
        )

        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(grading.answer_is_correct) if grading.answer_is_correct is not None else None,
                higher_is_better=True,
                llm_judge_prompt=grading.judge_prompt,
                llm_judge_response=grading.judge_response,
            )
        ]
