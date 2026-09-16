"""ARC (AI2 Reasoning Challenge): https://huggingface.co/datasets/allenai/ai2_arc

Grade-school science multiple-choice questions, split into the ARC-Easy and ARC-Challenge subjects. The
base task scores the full answer text (cloze); the OLMES variant shows the options as space-prefixed
lettered choices and scores the letters; the IDK variant lets the model abstain with "I do not know".
"""

from typing import Any, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import ClozeStyle, MCStyle, TaskStyler, answer_key_to_index

_IDK_PREAMBLE = (
    "Answer only if you are confident, since mistakes may be penalised, while correct answers receive points. "
    "It is acceptable to answer with 'I do not know' if you are unsure, and you will receive 0 points."
)


@final
class ArcReader(ChoiceReader):
    """Reads an ARC item: the question and its answer options, with the correct one at ``answerKey`` —
    a letter (A–E) or a 1-based number, both normalised to a 0-based index."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(
            raw_question=item["question"],
            choices=item["choices"]["text"],
            correct_index=answer_key_to_index(item["answerKey"]),
        )


def _arc_benchmark(id: str, styler: TaskStyler, dataset: DatasetPolicy | None = None) -> Benchmark:
    dataset_policy = dataset if dataset is not None else pinned_by_framework("allenai/ai2_arc")
    return ComposedBenchmark.choice(
        id=id,
        reader=ArcReader(),
        styler=styler,
        sample_split="test",
        fewshot_split="train",
        subjects=ListOfSubjects(["ARC-Easy", "ARC-Challenge"]),
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


def arc(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _arc_benchmark("ARC", ClozeStyle(question_prefix="Question: ", cue_text="Answer:"), dataset)


def arc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(question_prefix="Question: ", cue_text="Answer:", space_prefixed_labels=True)
    return _arc_benchmark("ARC_OLMES", styler, dataset)


def arc_idk(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = ClozeStyle(
        question_prefix="Question: ", cue_text="Answer:", initial_prompt=lambda _subject: _IDK_PREAMBLE
    ).with_abstention_option(" I do not know.")
    return _arc_benchmark("ARC_IDK", styler, dataset)


ARC_BENCHMARKS: list[Benchmark] = [arc(), arc_olmes(), arc_idk()]
