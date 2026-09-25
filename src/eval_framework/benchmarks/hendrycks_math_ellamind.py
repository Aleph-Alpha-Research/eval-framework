"""German Hendrycks Math (EllaMind), minerva-style — composed.

https://huggingface.co/datasets/ellamind/hendrycks-math-multilingual

The German counterpart of the Minerva-OLMES MATH tasks: ``Aufgabe:`` / ``Lösung:`` prompt markers, few-shot
demonstrations sampled from the dataset (single-paragraph solutions only, so the ``\\n\\n`` stop separates
blocks), German final-answer lines, ``minerva_de`` extraction and the German Minerva metrics.

- ``MATHMinervaDE_OLMES`` / ``MATHMinervaDE_OLMES_NONL``: free-form, differ only in stop sequences.
- ``MATHMinervaDE_BPB_OLMES``: bits-per-byte of the single gold solution (same prompt + few-shot).
"""

from typing import Any, final, override

from eval_framework.answer import ExtractFromCompletion, PickFromCandidates
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice, Generative
from eval_framework.fewshot import FewShot, FewshotExample, FunctionRenderer, SampleSplit
from eval_framework.metrics.completion.math_minerva_completion import (
    MathMinervaCompletionDE,
    MathMinervaCompletionRelaxedDE,
)
from eval_framework.metrics.completion.minerva_math_utils import extract_answers, normalized_gold_from_solution
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle

HENDRYCKS_MATH_DE_DATASET_PATH = "ellamind/hendrycks-math-multilingual"
_MINERVA_DE_MAX_TOKENS = 1024


def _minerva_de_prompt(item: dict[str, Any]) -> str:
    return "Aufgabe:\n" + item["problem"] + "\n\n"


def _minerva_de_answer_line(item: dict[str, Any]) -> str:
    return (
        f"Finale Antwort: Die finale Antwort lautet ${item['extracted_answer']}$. Ich hoffe, die Antwort ist korrekt."
    )


def _minerva_de_gold(item: dict[str, Any]) -> str:
    # "" reads the same as None to the minerva metric; keeps the gold a str (see math_reasoning._minerva_gold).
    return normalized_gold_from_solution(item["solution"]) or ""


def _minerva_de_extractor(completion_text: str) -> str:
    candidates = extract_answers(completion_text, use_cot=True, cot_style="minerva_de", relaxed=True)
    return candidates[0] if candidates else "[no_answer]"


def _generative_demo(item: dict[str, Any]) -> FewshotExample:
    # Fewshot answer = the "Lösung:" cue, the solution, then the German Minerva final-answer line.
    return FewshotExample(
        prompt=_minerva_de_prompt(item),
        answer=f"Lösung: {item['solution']}\n{_minerva_de_answer_line(item)}",
    )


def _single_paragraph_solution(example: dict[str, Any]) -> bool:
    # Keep only single-paragraph solutions: the "\n\n" stop must separate few-shot blocks, not fire inside one.
    return "\n\n" not in example["solution"]


def _de_dataset(dataset: DatasetPolicy | None) -> DatasetPolicy:
    return dataset if dataset is not None else pinned_by_framework(HENDRYCKS_MATH_DE_DATASET_PATH)


def _mathminerva_de(id: str, stop_sequences: list[str], dataset: DatasetPolicy | None) -> Benchmark:
    kind = Generative(
        build_prompt=_minerva_de_prompt,
        cue="Lösung:",
        ground_truth=_minerva_de_gold,
        metrics=[MathMinervaCompletionDE, MathMinervaCompletionRelaxedDE],
    )
    return ComposedBenchmark.compose(
        id=id,
        kind=kind,
        answer=ExtractFromCompletion(_minerva_de_extractor, stop_sequences, max_tokens=_MINERVA_DE_MAX_TOKENS),
        sample_split="test",
        fewshot=FewShot(SampleSplit(keep=_single_paragraph_solution), FunctionRenderer(_generative_demo)),
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=_de_dataset(dataset),
        language=Language.DEU,
    )


def mathminerva_de_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _mathminerva_de("MATHMinervaDE_OLMES", ["Aufgabe:", "\n\n"], dataset)


def mathminerva_de_olmes_nonl(dataset: DatasetPolicy | None = None) -> Benchmark:
    # Same as MATHMinervaDE_OLMES but stops only on "Aufgabe:" (drops the blank-line stop).
    return _mathminerva_de("MATHMinervaDE_OLMES_NONL", ["Aufgabe:"], dataset)


@final
class _MinervaDeBpbStyler(BPBStyle):
    """BPB scoring of the single gold solution, but under the German Minerva prompt (``Aufgabe:`` / ``Lösung:``,
    blank line before the cue) rather than the cloze ``question_prefix`` layout."""

    @override
    def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
        return f"Aufgabe:\n{raw_question}\n\n"


@final
class _MinervaDeBpbReader(ChoiceReader):
    """The single scored 'choice' is the gold solution plus the German final-answer line."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        choice = f"{item['solution']}\n{_minerva_de_answer_line(item)}"
        return ChoiceFields(raw_question=item["problem"], choices=[choice], correct_index=0)


def mathminerva_de_bpb_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    # Same prompt and few-shot as the generative variant; only the eval turn changes to a scored gold solution.
    return ComposedBenchmark.compose(
        id="MATHMinervaDE_BPB_OLMES",
        kind=Choice(_MinervaDeBpbReader(), _MinervaDeBpbStyler(cue_text="Lösung:")),
        answer=PickFromCandidates(),
        sample_split="test",
        fewshot=FewShot(SampleSplit(keep=_single_paragraph_solution), FunctionRenderer(_generative_demo)),
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=_de_dataset(dataset),
        language=Language.DEU,
    )


HENDRYCKS_MATH_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    mathminerva_de_olmes(),
    mathminerva_de_olmes_nonl(),
    mathminerva_de_bpb_olmes(),
]
