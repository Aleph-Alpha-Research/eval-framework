"""German HellaSwag (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/hellaswag-multilingual

HellaSwag is a sentence-completion task: the prompt is a partial sentence (``"{activity}: {context}"``)
and the model scores full-sentence endings. There is no natural MC variant. HellaSwag supplies separate
easy and hard distractors.
"""

from typing import Any, Literal, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, TaskStyler, shuffle_correct_with_distractors


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


def _hellaswag_ellamind_benchmark(
    id: str, styler: TaskStyler, distractor_level: Literal["easy", "hard"], dataset: DatasetPolicy | None
) -> Benchmark:
    return ComposedBenchmark.compose(
        id=id,
        styler=styler,
        reader=HellaswagReader(distractor_level),
        sample_split="validation",
        fewshot_split="validation",
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=dataset if dataset is not None else pinned_by_framework("ellamind/hellaswag-multilingual"),
        language=Language.DEU,
    )


# Sentence-completion: no question prefix, no cue, the continuation follows the context directly.
def _cloze_completion_style() -> ClozeStyle:
    return ClozeStyle(question_prefix="", trailing_newline=False, cue_text="")


def hellaswag_ellamind_easy_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _hellaswag_ellamind_benchmark("HELLASWAG_ELLAMIND_EASY_DE", _cloze_completion_style(), "easy", dataset)


def hellaswag_ellamind_hard_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _hellaswag_ellamind_benchmark("HELLASWAG_ELLAMIND_HARD_DE", _cloze_completion_style(), "hard", dataset)


def hellaswag_ellamind_bpb_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = BPBStyle(question_prefix="", trailing_newline=False, cue_text="")
    return _hellaswag_ellamind_benchmark("HELLASWAG_ELLAMIND_BPB_DE", styler, "easy", dataset)


HELLASWAG_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    hellaswag_ellamind_easy_de(),
    hellaswag_ellamind_hard_de(),
    hellaswag_ellamind_bpb_de(),
]
