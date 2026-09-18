"""How a benchmark obtains the model's scored answer.

An ``AnswerPolicy`` owns the "answer side" of a composed benchmark — the mode the model answers in
(loglikelihood over candidates vs. free-form completion), the bounds on any generation (stop sequences,
token limit), and how the raw generation is distilled into the answer that metrics score. It is injected
into ``compose`` next to the eval kind, so the kind stays purely about the prompt and candidates.
"""

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final, override

from eval_framework.contract import ResponseType
from eval_framework.metrics.efficiency.bytes_per_sequence_position import (
    BytesCompletion,
    BytesLoglikelihood,
    SequencePositionsCompletion,
    SequencePositionsLoglikelihood,
)
from eval_framework.metrics.efficiency.finish_reason import FinishReason
from eval_framework.metrics.efficiency.token_counters import TokenCounts
from eval_framework.shared.types import BaseMetricContext
from template_formatting.formatter import Message

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric


class AnswerPolicy(ABC):
    """The answer side of a kind: the response type, the generation bounds, and answer extraction."""

    @abstractmethod
    def response_type(self) -> ResponseType:
        """Whether the model is scored by loglikelihood over candidates or by free-form completion."""

    @abstractmethod
    def metrics(self) -> list[type["BaseMetric"]]:
        """The bookkeeping metrics this answer mode always reports (efficiency / token counts), added to the
        kind's scoring metrics."""

    @abstractmethod
    def stop_sequences(self) -> list[str]:
        """Stop sequences for completion generation (empty when nothing is generated)."""

    @abstractmethod
    def max_tokens(self) -> int | None:
        """Token limit for completion generation, or None for no limit."""

    @abstractmethod
    def extract_answer(
        self,
        completion_text: str,
        *,
        context: BaseMetricContext | list[BaseMetricContext] | None,
        ground_truth: str | list[str] | None,
        messages: list[Message],
    ) -> str:
        """The answer to score, distilled from the raw generation."""


@final
class PickFromCandidates(AnswerPolicy):
    """Loglikelihood scoring: the model is scored over fixed candidate completions and the answer is the
    best-scoring candidate, taken verbatim — nothing is generated, so nothing is bounded or extracted."""

    @override
    def response_type(self) -> ResponseType:
        return ResponseType.LOGLIKELIHOODS

    @override
    def metrics(self) -> list[type["BaseMetric"]]:
        return [BytesLoglikelihood, SequencePositionsLoglikelihood]

    @override
    def stop_sequences(self) -> list[str]:
        return []

    @override
    def max_tokens(self) -> int | None:
        return None

    @override
    def extract_answer(
        self,
        completion_text: str,
        *,
        context: BaseMetricContext | list[BaseMetricContext] | None,
        ground_truth: str | list[str] | None,
        messages: list[Message],
    ) -> str:
        return completion_text


@final
class ExtractFromCompletion(AnswerPolicy):
    """Free-form completion: the model generates (bounded by ``stop_sequences`` / ``max_tokens``) and the
    scored answer is pulled out with ``answer_re``. ``last_match`` takes the final match, upper-cased (for
    lenient case-insensitive patterns); otherwise the first match is returned as-is. ``"[invalid]"`` when
    nothing matches."""

    def __init__(
        self,
        answer_re: re.Pattern[str],
        stop_sequences: list[str] | None = None,
        *,
        last_match: bool = False,
        max_tokens: int | None = None,
    ) -> None:
        self._answer_re = answer_re
        self._stop_sequences = list(stop_sequences) if stop_sequences is not None else []
        self._last_match = last_match
        self._max_tokens = max_tokens

    @override
    def response_type(self) -> ResponseType:
        return ResponseType.COMPLETION

    @override
    def metrics(self) -> list[type["BaseMetric"]]:
        return [BytesCompletion, SequencePositionsCompletion, TokenCounts, FinishReason]

    @override
    def stop_sequences(self) -> list[str]:
        return list(self._stop_sequences)

    @override
    def max_tokens(self) -> int | None:
        return self._max_tokens

    @override
    def extract_answer(
        self,
        completion_text: str,
        *,
        context: BaseMetricContext | list[BaseMetricContext] | None,
        ground_truth: str | list[str] | None,
        messages: list[Message],
    ) -> str:
        if self._last_match:
            matches = self._answer_re.findall(completion_text)
            return matches[-1].upper() if matches else "[invalid]"
        match = self._answer_re.search(completion_text)
        return match.group(1) if match else "[invalid]"
