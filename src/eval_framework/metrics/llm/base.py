from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import BaseMetric
from eval_framework.shared.types import Completion


class BaseLLMJudgeMetric(BaseMetric[Completion]):
    def __init__(self, llm_judge: BaseLLM) -> None:
        self._llm_judge = llm_judge
