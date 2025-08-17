import json
from enum import Enum

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.metrics.completion_metrics.text_counter import ParagraphCounter, SentenceCounter, WordCounter
from eval_framework.shared.types import Completion


class LengthRequirementUnit(Enum):
    WORDS = "words"
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"


class LengthRequirementType(Enum):
    MIN = "minimum"
    MAX = "maximum"
    TARGET = "target"


class LengthControl(BaseMetric[Completion]):
    NAME = "length_control"

    def __init__(self, tolerance: float = 1 / 6) -> None:
        super().__init__()
        self.tolerance = tolerance

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [
                MetricResult(
                    metric_name=f"{self.NAME}/fulfills_length_requirement",
                    value=None,
                    higher_is_better=True,
                    error=response.error if response.error is not None else None,
                )
            ]

        expectations = json.loads(str(response.ground_truth))
        stripped_completion = response.completion.strip()

        match LengthRequirementUnit(expectations["unit"]):
            case LengthRequirementUnit.WORDS:
                count = WordCounter._count_words(stripped_completion)
            case LengthRequirementUnit.SENTENCES:
                count = SentenceCounter._count_sentences(stripped_completion)
            case LengthRequirementUnit.PARAGRAPHS:
                count = ParagraphCounter._count_paragraphs(stripped_completion)
            case _:
                raise NotImplementedError(f"LengthRequirementUnit {expectations['unit']} is not supported.")

        expected_count = int(expectations["count"])
        normalized_distance_to_target = (count - expected_count) / float(expected_count)
        absolute_normalized_distance_to_target = abs(normalized_distance_to_target)

        match LengthRequirementType(expectations["type"]):
            case LengthRequirementType.TARGET:
                fulfills_length_requirement = absolute_normalized_distance_to_target <= self.tolerance
            case LengthRequirementType.MIN:
                fulfills_length_requirement = count >= expected_count
            case LengthRequirementType.MAX:
                fulfills_length_requirement = count <= expected_count
            case _:
                raise NotImplementedError(f"LengthRequirementType {expectations['type']} is not supported.")

        return [
            MetricResult(
                metric_name=f"{self.NAME}/normalized_distance_to_target",
                value=float(normalized_distance_to_target),
                higher_is_better=False,
            ),
            MetricResult(
                metric_name=f"{self.NAME}/absolute_normalized_distance_to_target",
                value=float(absolute_normalized_distance_to_target),
                higher_is_better=False,
            ),
            MetricResult(
                metric_name=f"{self.NAME}/fulfills_length_requirement",
                value=float(fulfills_length_requirement),
                higher_is_better=True,
                error=response.error,
            ),
        ]
