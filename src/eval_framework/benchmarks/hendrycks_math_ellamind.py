from dataclasses import dataclass
from typing import Any, Literal, final, get_args, override

from eval_framework.answer import AnswerPolicy, ExtractFromCompletion, PickFromCandidates
from eval_framework.benchmarks.utils.ellamind_languages import ellamind_language
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice, EvalKind, Generative
from eval_framework.fewshot import FewShot, FewshotExample, FunctionRenderer, SampleSplit
from eval_framework.metrics.base import BaseMetric
from eval_framework.metrics.completion.math_minerva_completion import (
    MathMinervaCompletionDE,
    MathMinervaCompletionRelaxedDE,
)
from eval_framework.metrics.completion.minerva_math_utils import extract_answers, normalized_gold_from_solution
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle

HENDRYCKS_MATH_ELLAMIND_DATASET_PATH = "ellamind/hendrycks-math-multilingual"
_MAX_TOKENS = 1024

TaskVariant = Literal["OLMES", "OLMES_NONL", "BPB"]

LanguageCode = Literal["DE"]
_LANGUAGES: tuple[LanguageCode, ...] = get_args(LanguageCode)

# Variant -> task-id template. The ids predate the naming convention and are kept for continuity.
_TASK_ID_TEMPLATES: dict[TaskVariant, str] = {
    "OLMES": "MATHMinerva{language}_OLMES",
    "OLMES_NONL": "MATHMinerva{language}_OLMES_NONL",
    "BPB": "MATHMinerva{language}_BPB_OLMES",
}


@dataclass(frozen=True, kw_only=True)
class MinervaLocale:
    """One language's wording of the Minerva prompt, and the extraction and metrics that understand it."""

    problem_prefix: str
    """Opens the problem block (``Problem:``); also the stop sequence of a free-form generation."""
    solution_cue: str
    """Primes the solution turn (``Solution:``)."""
    final_answer_line: str
    """Closes a shown solution; ``{answer}`` is the item's extracted answer. Its wording must be what the
    language's ``cot_style`` extractor parses."""
    cot_style: str
    """Key into ``minerva_math_utils.COT_EXTRACTORS``."""
    metrics: list[type[BaseMetric]]
    """The strict and relaxed Minerva metrics for ``cot_style``."""

    def __post_init__(self) -> None:
        if "{answer}" not in self.final_answer_line:
            raise ValueError(f"final_answer_line must contain '{{answer}}': {self.final_answer_line!r}")


_LOCALES: dict[LanguageCode, MinervaLocale] = {
    "DE": MinervaLocale(
        problem_prefix="Aufgabe:",
        solution_cue="Lösung:",
        final_answer_line="Finale Antwort: Die finale Antwort lautet ${answer}$. Ich hoffe, die Antwort ist korrekt.",
        cot_style="minerva_de",
        metrics=[MathMinervaCompletionDE, MathMinervaCompletionRelaxedDE],
    ),
}


@final
class _MinervaPrompt:
    """Assembles the Minerva prompt pieces in one language — shared by the generative and BPB variants."""

    def __init__(self, locale: MinervaLocale) -> None:
        self._locale = locale

    def problem(self, item: dict[str, Any]) -> str:
        return f"{self._locale.problem_prefix}\n{item['problem']}\n\n"

    def solution(self, item: dict[str, Any]) -> str:
        """The gold solution closed by the final-answer line: the BPB target and the body of a demonstration."""
        return f"{item['solution']}\n{self._locale.final_answer_line.format(answer=item['extracted_answer'])}"

    def demo(self, item: dict[str, Any]) -> FewshotExample:
        return FewshotExample(prompt=self.problem(item), answer=f"{self._locale.solution_cue} {self.solution(item)}")

    def extract(self, completion_text: str) -> str:
        candidates = extract_answers(completion_text, use_cot=True, cot_style=self._locale.cot_style, relaxed=True)
        return candidates[0] if candidates else "[no_answer]"


def _minerva_gold(item: dict[str, Any]) -> str:
    # "" reads the same as None to the minerva metric; keeps the gold a str (see math_reasoning._minerva_gold).
    return normalized_gold_from_solution(item["solution"]) or ""


def _single_paragraph_solution(example: dict[str, Any]) -> bool:
    # Keep only single-paragraph solutions: the "\n\n" stop must separate few-shot blocks, not fire inside one.
    return "\n\n" not in example["solution"]


@final
class _MinervaBpbStyler(BPBStyle):
    """BPB scoring of the single gold solution, but under the Minerva prompt (problem prefix on its own line,
    blank line before the cue) rather than the cloze ``question_prefix`` layout."""

    def __init__(self, locale: MinervaLocale) -> None:
        super().__init__(cue_text=locale.solution_cue)
        self._problem_prefix = locale.problem_prefix

    @override
    def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
        return f"{self._problem_prefix}\n{raw_question}\n\n"


@final
class _MinervaBpbReader(ChoiceReader):
    """The single scored 'choice' is the gold solution plus the language's final-answer line."""

    def __init__(self, prompt: _MinervaPrompt) -> None:
        self._prompt = prompt

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(raw_question=item["problem"], choices=[self._prompt.solution(item)], correct_index=0)


def hendrycks_math_ellamind(
    task_variant: TaskVariant, language: LanguageCode, dataset: DatasetPolicy | None = None
) -> Benchmark:
    """Build ``MATHMinerva{language}_{variant}``, e.g. ``hendrycks_math_ellamind("OLMES", "DE")``."""
    locale = _LOCALES[language]
    prompt = _MinervaPrompt(locale)
    lang, config = ellamind_language(language)

    kind: EvalKind
    answer: AnswerPolicy
    if task_variant == "BPB":
        # Same prompt and few-shot as the generative variants; only the eval turn changes to a scored gold solution.
        kind = Choice(_MinervaBpbReader(prompt), _MinervaBpbStyler(locale))
        answer = PickFromCandidates()
    else:
        # OLMES stops at the next problem or a blank line; OLMES_NONL drops the blank-line stop.
        stop_sequences = [locale.problem_prefix] if task_variant == "OLMES_NONL" else [locale.problem_prefix, "\n\n"]
        kind = Generative(
            build_prompt=prompt.problem,
            cue=locale.solution_cue,
            ground_truth=_minerva_gold,
            metrics=locale.metrics,
        )
        answer = ExtractFromCompletion(prompt.extract, stop_sequences, max_tokens=_MAX_TOKENS)

    return ComposedBenchmark.compose(
        id=_TASK_ID_TEMPLATES[task_variant].format(language=language),
        kind=kind,
        answer=answer,
        sample_split="test",
        fewshot=FewShot(SampleSplit(keep=_single_paragraph_solution), FunctionRenderer(prompt.demo)),
        subjects=ListOfSubjects([config]),
        dataset_policy=dataset if dataset is not None else pinned_by_framework(HENDRYCKS_MATH_ELLAMIND_DATASET_PATH),
        language=lang,
    )


HENDRYCKS_MATH_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    hendrycks_math_ellamind(task_variant, language) for language in _LANGUAGES for task_variant in get_args(TaskVariant)
]
