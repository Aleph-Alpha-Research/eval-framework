"""SciQ (English): https://huggingface.co/datasets/allenai/sciq

Science-exam questions, each with three distractors and one correct answer. The four options are shuffled
deterministically per item (seeded by the question + answer). The registered variant is OLMES-style: the
options are shown as space-prefixed lettered choices (" A. …") and the model is scored over the letters.
"""

from typing import Any, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import MCStyle, shuffle_correct_with_distractors


@final
class SciqReader(ChoiceReader):
    """Reads a SciQ item: the question and its options — the correct answer shuffled among its three
    distractors, deterministically per item (seeded by question + answer)."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        choices, correct_index = shuffle_correct_with_distractors(
            correct=item["correct_answer"],
            distractors=[item["distractor1"], item["distractor2"], item["distractor3"]],
            seed_text=item["question"] + item["correct_answer"],
        )
        return ChoiceFields(raw_question=item["question"], choices=choices, correct_index=correct_index)


def sciq_mc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(question_prefix="Question: ", cue_text="Answer:", space_prefixed_labels=True)
    dataset_policy = dataset if dataset is not None else pinned_by_framework("allenai/sciq")
    # The registry/hash identity is the compact "SCIQ_OLMES"; the display name keeps the SciQ casing.
    return ComposedBenchmark.choice(
        id="SCIQ_OLMES",
        display_name="SciQ_OLMES",
        reader=SciqReader(),
        styler=styler,
        sample_split="train",
        fewshot_split="train",
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


SCIQ_BENCHMARKS: list[Benchmark] = [sciq_mc_olmes()]
