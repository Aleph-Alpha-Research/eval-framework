"""ARC German (ARC-DE)."""

from typing import Any

from eval_framework.composed import ChoiceFields, ChoiceReader, ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.tasks.base import NO_SUBJECT, Language
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import ClozeStyle, answer_key_to_index


class ArcDeReader(ChoiceReader):
    """Reads an ARC-DE row into choice fields: the German question, its answer texts, and the correct index.

    ``answerKey`` arrives as either a 1-based number or a letter; ``answer_key_to_index`` normalises both,
    which also frees the reader from caring how many answers a given row offers.
    """

    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(
            raw_question=item["question_de"],
            choices=item["choices_de"]["text"],
            correct_index=answer_key_to_index(item["answerKey"]),
        )


def arc_de_benchmark() -> Benchmark:
    """ARC-DE as cloze/ranked classification

    https://huggingface.co/datasets/LeoLM/ArcChallenge_de
    """
    return ComposedBenchmark.compose(
        id="ARC_DE",
        display_name="ARC German",
        styler=ClozeStyle(question_prefix="Frage: ", cue_text="Antwort:"),
        reader=ArcDeReader(),
        sample_split="test",
        fewshot_split="validation",
        subjects=[NO_SUBJECT],
        dataset_policy=pinned_by_framework("LeoLM/ArcChallenge_de"),
        language=Language.DEU,
    )
