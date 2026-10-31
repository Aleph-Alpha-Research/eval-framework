"""Multilingual PIQA (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/piqa-multilingual

PIQA supplies separate easy and hard distractors. Each language is one config of the dataset (``deu``). The
prompt opens with a language-specific "Goal:" prefix (PIQA's ``goal`` field is a goal, not a question) and
closes with the framework's default answer cue for the language.
"""

from typing import Any, Literal, final, get_args, override

from eval_framework.benchmarks.utils.ellamind_languages import ellamind_language
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, TaskStyler, shuffle_correct_with_distractors

TaskVariant = Literal["CLOZE_EASY", "CLOZE_HARD", "MC_EASY", "MC_HARD", "BPB"]

LanguageCode = Literal["DE", "FR"]
_LANGUAGES: tuple[LanguageCode, ...] = get_args(LanguageCode)

# Variant -> (styler class, distractor level). BPB scores only the gold continuation, so its distractor level
# is irrelevant; ``easy`` is used.
_VARIANTS: dict[TaskVariant, tuple[type[TaskStyler], Literal["easy", "hard"]]] = {
    "CLOZE_EASY": (ClozeStyle, "easy"),
    "CLOZE_HARD": (ClozeStyle, "hard"),
    "MC_EASY": (MCStyle, "easy"),
    "MC_HARD": (MCStyle, "hard"),
    "BPB": (BPBStyle, "easy"),
}

# PIQA's prompt opens with the goal, so the usual "Question:" prefix is replaced by a per-language "Goal:".
_GOAL_PREFIX: dict[LanguageCode, str] = {"DE": "Ziel: ", "FR": "Objectif : "}


@final
class PiqaReader(ChoiceReader):
    """Reads a PIQA item: a single easy/hard distractor per level, shuffled in with the correct solution."""

    def __init__(self, distractor_level: Literal["easy", "hard"]) -> None:
        self._distractor_level = distractor_level

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        distractor = item["easy_distractor"] if self._distractor_level == "easy" else item["hard_distractor"]
        choices, correct_index = shuffle_correct_with_distractors(
            correct=item["correct_solution"],
            distractors=[distractor],
            seed_text=item["goal"] + item["correct_solution"],
        )
        return ChoiceFields(raw_question=item["goal"], choices=choices, correct_index=correct_index)


def piqa_ellamind(task_variant: TaskVariant, language: LanguageCode, dataset: DatasetPolicy | None = None) -> Benchmark:
    """Build ``PIQA_ELLAMIND_{variant}_{language}``, e.g. ``piqa_ellamind("MC_EASY", "DE")``."""
    styler_cls, distractor_level = _VARIANTS[task_variant]
    lang, config = ellamind_language(language)
    dataset_policy = dataset if dataset is not None else pinned_by_framework("ellamind/piqa-multilingual")
    return ComposedBenchmark.choice(
        id=f"PIQA_ELLAMIND_{task_variant}_{language}",
        reader=PiqaReader(distractor_level),
        styler=styler_cls.for_language(lang, question_prefix=_GOAL_PREFIX[language]),
        sample_split="validation",
        fewshot_split="validation",
        subjects=ListOfSubjects([config]),
        dataset_policy=dataset_policy,
        language=lang,
    )


PIQA_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    piqa_ellamind(task_variant, language) for language in _LANGUAGES for task_variant in _VARIANTS
]
