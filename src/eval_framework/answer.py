"""How a benchmark obtains the model's scored answer.

An ``AnswerPolicy`` owns the "answer side" of a composed benchmark — the mode the model answers in
(loglikelihood over candidates vs. free-form completion), the bounds on any generation (stop sequences,
token limit), and how the raw generation is distilled into the answer that metrics score. It is injected
into ``compose`` next to the eval kind, so the kind stays purely about the prompt and candidates.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, final, override

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


# A generation -> the scored answer distilled from it ("[invalid]" when none can be).
Extractor = Callable[[str], str]


def first_match(answer_re: re.Pattern[str]) -> Extractor:
    """Extractor: group 1 of the first regex match, returned as-is, or ``"[invalid]"``."""

    def extract(completion_text: str) -> str:
        match = answer_re.search(completion_text)
        return match.group(1) if match else "[invalid]"

    return extract


def last_match(answer_re: re.Pattern[str]) -> Extractor:
    """Extractor: the last regex match, upper-cased (for lenient case-insensitive patterns), or
    ``"[invalid]"``."""

    def extract(completion_text: str) -> str:
        matches = answer_re.findall(completion_text)
        return matches[-1].upper() if matches else "[invalid]"

    return extract


@final
class ExtractFromCompletion(AnswerPolicy):
    """Free-form completion: the model generates (bounded by ``stop_sequences`` / ``max_tokens``) and the
    scored answer is produced by ``extract`` applied to the generation. Regex extractors are available as
    ``first_match`` / ``last_match``."""

    def __init__(
        self,
        extract: Extractor,
        stop_sequences: list[str] | None = None,
        *,
        max_tokens: int | None = None,
    ) -> None:
        self._extract = extract
        self._stop_sequences = stop_sequences or []
        self._max_tokens = max_tokens

    @override
    def response_type(self) -> ResponseType:
        return ResponseType.COMPLETION

    @override
    def metrics(self) -> list[type["BaseMetric"]]:
        return [BytesCompletion, SequencePositionsCompletion, TokenCounts, FinishReason]

    @override
    def stop_sequences(self) -> list[str]:
        return self._stop_sequences

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
        return self._extract(completion_text)


class CodeReconstructor(Protocol):
    """Assembles the runnable program a code-execution metric runs, out of the raw generation plus the
    sample's scoring material — the test harness / code prompt carried in ``context`` and the gold asserts
    in ``ground_truth``. This is the code-generation counterpart of ``ExtractFromCompletion``'s extractor,
    but it needs more than the generation text: the snippet only becomes runnable once spliced together with
    the problem's tests."""

    def __call__(
        self,
        completion_text: str,
        *,
        context: BaseMetricContext | list[BaseMetricContext] | None,
        ground_truth: str | list[str] | None,
        messages: list[Message],
    ) -> str: ...


@final
class ReconstructProgram(AnswerPolicy):
    """Free-form code generation scored by execution: the model generates a solution (bounded by
    ``stop_sequences`` / ``max_tokens``) and ``reconstruct`` turns it into the runnable program the metric
    executes — typically the generated snippet spliced into the prompt and test harness from the sample's
    context. The reconstructed program *is* the scored answer, so the executing metric runs it verbatim."""

    def __init__(
        self,
        reconstruct: CodeReconstructor,
        *,
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._reconstruct = reconstruct
        self._stop_sequences = stop_sequences or []
        self._max_tokens = max_tokens

    @override
    def response_type(self) -> ResponseType:
        return ResponseType.COMPLETION

    @override
    def metrics(self) -> list[type["BaseMetric"]]:
        return [BytesCompletion, SequencePositionsCompletion, TokenCounts, FinishReason]

    @override
    def stop_sequences(self) -> list[str]:
        return self._stop_sequences

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
        return self._reconstruct(completion_text, context=context, ground_truth=ground_truth, messages=messages)
