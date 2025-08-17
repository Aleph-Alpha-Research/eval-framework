import re
from collections import Counter
from typing import Final, Sequence

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class WordRepetition(BaseMetric[Completion]):
    """Word Repetition Metric

    This metric checks for repetitions of words in the completion text for a
    given window size and repetition threshold. The window size defines the
    consecutive word count to consider a repetition, and min_repetitions
    specifies the minimum repetition count that triggers the metric. This metric
    returns 0.0 if no repetitions are found, and 1.0 if a sufficient number of
    repetitions are found. For example, if the completion contains a two-word
    sequence that repeats once (such as "hello world hello world"), this metric
    would trigger with a window size of 2 and min_repetitions set to 1.
    """

    NAME = "WordRepetition"
    HIGHER_IS_BETTER: Final[bool] = False

    def __init__(self, window_size: int = 128, min_repetitions: int = 1) -> None:
        """
        Initialize the WordRepetition metric.

        Args:
            window_size (int): The number of consecutive words to consider as a
                sequence.
            min_repetitions (int): The minimum number of times a sequence must
                repeat to be considered a repetition. Set to 1 to catch any
                repetition.
        """
        super().__init__()
        self.window_size = window_size
        self.min_repetitions = min_repetitions

        if self.min_repetitions < 1:
            raise ValueError("min_repetitions must be at least 1")

        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [
                MetricResult(
                    metric_name=self.NAME,
                    value=None,
                    higher_is_better=self.HIGHER_IS_BETTER,
                    error=response.error,
                )
            ]

        has_repetition = _has_repetition(
            text=response.completion,
            window_size=self.window_size,
            min_repetitions=self.min_repetitions,
        )

        return [
            MetricResult(
                metric_name=self.NAME,
                value=float(has_repetition),
                higher_is_better=self.HIGHER_IS_BETTER,
                error=response.error,
            )
        ]


def _has_repetition(text: str, window_size: int, min_repetitions: int) -> bool:
    """Check if the text contains any word sequences of a given size that repeat"""
    sequences = _word_sequences(_to_words(text), window_size)
    counts = Counter(sequences)
    return any([count > min_repetitions for count in counts.values()])


def _to_words(text: str) -> Sequence[str]:
    """A somewhat crude function to tokenize a string into words."""
    return re.findall(r"\b\w+\b", text, re.UNICODE)


def _word_sequences(text_words: Sequence[str], window_size: int) -> Sequence[Sequence[str]]:
    """Get all contiguous sub-sequences of a given size from a word sequence."""
    return [tuple(text_words[i : i + window_size]) for i in range(len(text_words) - window_size + 1)]
