"""Shared chain-of-thought scaffolding for multiple-choice benchmarks.

A CoT variant asks the model to reason freely and conclude with a stated answer letter, which is pulled back
out at scoring time (the injected ``ExtractFromCompletion``). The prompt surface — an optional preamble, the body,
and any inert scored candidates — is injected per benchmark; the CoT contract is fixed here: one free-form
sample, no assistant cue, the bare answer letter as ground truth, scored by accuracy.
"""

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, final, override

from eval_framework.answer import ExtractFromCompletion, first_match, last_match
from eval_framework.choices import ChoiceReader
from eval_framework.eval_kind import EvalKind, SampleBody, assemble_messages
from eval_framework.fewshot import FewshotExample
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.tasks.task_style import InitialPrompt
from eval_framework.tasks.utils import get_n_letters
from template_formatting.formatter import Message

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric

# (raw question, in-order choices) -> the prompt body shown to the model.
CotPrompt = Callable[[str, list[str]], str]
# (this item's answer-letter keys) -> inert scored candidates carried on the sample (usually none).
CotCandidates = Callable[[list[str]], list[str]]


def tulu3_cot_prompt(raw_question: str, choices: list[str]) -> str:
    """The parenthesised-option CoT body shared by MMLU-Pro and GPQA.

    Reasoning prompt from Figure 44 of the Tülu 3 paper: https://arxiv.org/pdf/2411.15124
    """
    keys = get_n_letters(len(choices))
    options = "\n".join(f"({key}) {choice}" for key, choice in zip(keys, choices))
    return (
        "Answer the following multiple-choice question by giving the correct answer letter in parentheses. "
        "Provide CONCISE reasoning for the answer, and make sure to finish the response with "
        '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.'
        f"\n\nQuestion: {raw_question}\n{options}"
        "\n\nAnswer the above question and REMEMBER to finish your response with the exact phrase "
        '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.'
    )


def tulu_answer() -> ExtractFromCompletion:
    """Extracts the parenthesised letter that ``tulu3_cot_prompt`` asks the model to conclude with —
    ``"Therefore, the answer is (X)"``. Kept here beside the prompt because both encode the same ``(X)``
    format. The accepted letters are the fixed A–J the prompt's ``"(A), (B), ..., (E), etc."`` implies."""
    return ExtractFromCompletion(first_match(re.compile(r"Therefore, the answer is \(([ABCDEFGHIJ])\)")), ["Question:"])


def tulu_answer_v2(n_options: int) -> ExtractFromCompletion:
    """Lenient variant of ``tulu_answer``: no required ``"Therefore,"``, optional colon and parentheses,
    case-insensitive, taking the last match. Only the accepted letter range is benchmark-specific, so it is
    built from ``n_options``."""
    letters = "".join(get_n_letters(n_options))
    return ExtractFromCompletion(last_match(re.compile(rf"\banswer\s+is:?\s*\(?([{letters}])\b\)?", re.IGNORECASE)))


@final
class Cot(EvalKind):
    """Multiple-choice chain-of-thought (see the module docstring). ``build_prompt`` renders the body,
    ``preamble`` an optional subject-templated top line, and ``candidates`` any inert scored letters kept for
    faithful parity with a loglikelihood baseline (free-form scoring ignores them)."""

    def __init__(
        self,
        reader: ChoiceReader,
        *,
        build_prompt: CotPrompt,
        preamble: InitialPrompt | None = None,
        candidates: CotCandidates | None = None,
    ) -> None:
        self._reader = reader
        self._build_prompt = build_prompt
        self._preamble = preamble
        self._candidates = candidates

    @override
    def metrics(self) -> list[type["BaseMetric"]]:
        return [AccuracyCompletion]

    @override
    def messages(self, body: SampleBody, *, fewshot: list[FewshotExample], subject_label: str) -> list[Message]:
        preamble = self._preamble(subject_label) if self._preamble is not None else None
        return assemble_messages(fewshot, body, initial_prompt=preamble)

    @override
    def samples(self, item: dict[str, Any]) -> list[SampleBody]:
        fields = self._reader.read(item)
        keys = get_n_letters(len(fields.choices))
        return [
            SampleBody(
                prompt=self._build_prompt(fields.raw_question, fields.choices),
                cue="",  # no assistant cue — the model continues into its own reasoning
                possible_completions=self._candidates(keys) if self._candidates is not None else [],
                ground_truth=keys[fields.correct_index],  # the bare answer letter
            )
        ]
