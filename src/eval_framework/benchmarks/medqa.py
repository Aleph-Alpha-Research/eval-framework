"""MedQA (English): https://huggingface.co/datasets/davidheineman/medqa-en

Multiple-choice questions from medical licensing exams. The registered variant is OLMES-style: the options
are shown as space-prefixed lettered choices (" A. …") and the model is scored over the letter labels.
"""

from typing import Any, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import MCStyle


@final
class MedqaReader(ChoiceReader):
    """Reads a MedQA item: the exam question and its options, with the correct one at ``answer_idx``."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(
            raw_question=item["question"],
            choices=item["choices"],
            correct_index=int(item["answer_idx"]),
        )


def medqa_mc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(question_prefix="Question: ", cue_text="Answer:", space_prefixed_labels=True)
    dataset_policy = dataset if dataset is not None else pinned_by_framework("davidheineman/medqa-en")
    return ComposedBenchmark.choice(
        id="MedQAMC_OLMES",
        reader=MedqaReader(),
        styler=styler,
        sample_split="test",
        fewshot_split="train",
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


MEDQA_BENCHMARKS: list[Benchmark] = [medqa_mc_olmes()]
