"""Multilingual ARC (EllaMind): https://huggingface.co/datasets/ellamind/arc-multilingual

Grade-school science questions translated into German, French, Spanish and Italian. Every slice of a language
loads that language's single config (``deu``, ``fra``, ``spa`` or ``ita``); the ARC-Easy and ARC-Challenge
subjects are the rows of that config tagged by the ``arc_config`` column. Cloze scores the full answer text,
MC the letter labels, BPB only the ground-truth answer's bits-per-byte.
"""

from typing import Any, Literal, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, TaskStyler, answer_key_to_index

Style = Literal["CLOZE", "MC", "BPB"]
LanguageCode = Literal["DE", "FR", "ES", "IT"]

# Style -> styler, and language code -> (language, dataset config).
_STYLES: dict[Style, type[TaskStyler]] = {"CLOZE": ClozeStyle, "MC": MCStyle, "BPB": BPBStyle}
_LANGUAGES: dict[LanguageCode, tuple[Language, str]] = {
    "DE": (Language.DEU, "deu"),
    "FR": (Language.FRA, "fra"),
    "ES": (Language.SPA, "spa"),
    "IT": (Language.ITA, "ita"),
}


@final
class ArcEllamindReader(ChoiceReader):
    """Reads a translated ARC item: the question and its answer options, with the correct one at ``answer_key``
    (a letter A–E or a 1-based number)."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(
            raw_question=item["question"],
            choices=item["choices"],
            correct_index=answer_key_to_index(item["answer_key"]),
        )


def arc_ellamind(style: Style, language: LanguageCode, dataset: DatasetPolicy | None = None) -> Benchmark:
    """Build ``ARC_ELLAMIND_{style}_{language}``, e.g. ``arc_ellamind("CLOZE", "DE")``."""
    lang, config = _LANGUAGES[language]
    # ARC-Easy and ARC-Challenge share the language's single config, tagged by the ``arc_config`` column.
    source = (
        dataset
        if dataset is not None
        else pinned_by_framework("ellamind/arc-multilingual").subject_encoded_in_column(
            config=config, column="arc_config"
        )
    )
    return ComposedBenchmark.choice(
        id=f"ARC_ELLAMIND_{style}_{language}",
        reader=ArcEllamindReader(),
        styler=_STYLES[style].for_language(lang),
        sample_split="test",
        fewshot_split="test",
        subjects=ListOfSubjects(["ARC-Easy", "ARC-Challenge"]),
        dataset_policy=source,
        language=lang,
    )


ARC_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    arc_ellamind(style, language) for language in _LANGUAGES for style in _STYLES
]
