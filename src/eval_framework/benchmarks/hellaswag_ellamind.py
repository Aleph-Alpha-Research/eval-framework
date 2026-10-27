"""Multilingual HellaSwag (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/hellaswag-multilingual

HellaSwag is a sentence-completion task: the prompt is a partial sentence (``"{activity}: {context}"``)
and the model scores full-sentence endings. There is no natural MC variant. HellaSwag supplies separate
easy and hard distractors. Each language is one config of the dataset (``deu``, ``fra``, ``spa``, ``ita``).
"""

from typing import Any, Literal, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, shuffle_correct_with_distractors

Variant = Literal["EASY", "HARD", "BPB"]
LanguageCode = Literal["DE", "FR", "ES", "IT"]

# Variant -> (styler class, distractor level). Sentence-completion styling: no question prefix, no cue, the
# continuation follows the context directly.
_VARIANTS: dict[Variant, tuple[type[ClozeStyle] | type[BPBStyle], Literal["easy", "hard"]]] = {
    "EASY": (ClozeStyle, "easy"),
    "HARD": (ClozeStyle, "hard"),
    "BPB": (BPBStyle, "easy"),
}
# Language code -> (language, dataset config).
_LANGUAGES: dict[LanguageCode, tuple[Language, str]] = {
    "DE": (Language.DEU, "deu"),
    "FR": (Language.FRA, "fra"),
    "ES": (Language.SPA, "spa"),
    "IT": (Language.ITA, "ita"),
}


@final
class HellaswagReader(ChoiceReader):
    """Reads a HellaSwag item: the partial sentence ``"{activity}: {context}"``, with the easy/hard
    full-sentence endings for the level shuffled in with the correct ending."""

    def __init__(self, distractor_level: Literal["easy", "hard"]) -> None:
        self._distractor_level = distractor_level

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        distractors = item["easy_distractors"] if self._distractor_level == "easy" else item["hard_distractors"]
        choices, correct_index = shuffle_correct_with_distractors(
            correct=item["correct_ending"],
            distractors=distractors,
            seed_text=item["context"] + item["correct_ending"],
        )
        return ChoiceFields(
            raw_question=f"{item['activity'].strip()}: {item['context'].strip()}",
            choices=choices,
            correct_index=correct_index,
        )


def hellaswag_ellamind(variant: Variant, language: LanguageCode, dataset: DatasetPolicy | None = None) -> Benchmark:
    styler_cls, distractor_level = _VARIANTS[variant]
    lang, config = _LANGUAGES[language]
    dataset_policy = dataset if dataset is not None else pinned_by_framework("ellamind/hellaswag-multilingual")
    return ComposedBenchmark.choice(
        id=f"HELLASWAG_ELLAMIND_{variant}_{language}",
        reader=HellaswagReader(distractor_level),
        styler=styler_cls(question_prefix="", trailing_newline=False, cue_text=""),
        sample_split="validation",
        fewshot_split="validation",
        subjects=ListOfSubjects([config]),
        dataset_policy=dataset_policy,
        language=lang,
    )


HELLASWAG_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    hellaswag_ellamind(variant, language) for language in _LANGUAGES for variant in _VARIANTS
]
