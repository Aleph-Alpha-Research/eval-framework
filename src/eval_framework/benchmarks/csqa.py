"""CommonsenseQA (English): https://huggingface.co/datasets/tau/commonsense_qa

Five-way multiple-choice commonsense questions. The registered variant is OLMES-style: the options are
shown as space-prefixed lettered choices (" A. …") and the model is scored over the letter labels.
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
class CommonsenseQaReader(ChoiceReader):
    """Reads a CommonsenseQA item: the question and its labelled options, with the correct one at ``answerKey``."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        labels = item["choices"]["label"]
        return ChoiceFields(
            raw_question=item["question"],
            choices=item["choices"]["text"],
            correct_index=labels.index(item["answerKey"]),
        )


def csqa_mc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(question_prefix="Question: ", cue_text="Answer:", space_prefixed_labels=True)
    dataset_policy = dataset if dataset is not None else pinned_by_framework("tau/commonsense_qa")
    return ComposedBenchmark.choice(
        id="CommonsenseQAMC_OLMES",
        reader=CommonsenseQaReader(),
        styler=styler,
        sample_split="train",
        fewshot_split="train",
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


CSQA_BENCHMARKS: list[Benchmark] = [csqa_mc_olmes()]
