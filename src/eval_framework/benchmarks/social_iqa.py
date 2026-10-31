"""Social IQa (English): https://huggingface.co/datasets/allenai/social_i_qa

Commonsense reasoning about social situations: a context and question with three answer options
(``answerA``/``answerB``/``answerC``) and a 1-indexed ``label`` marking the correct one. The registered
variant is OLMES-style: options are shown as space-prefixed lettered choices (" A. …") and the model is
scored over the letter labels.

``allenai/social_i_qa`` ships only a (no-longer-supported) loading script on its main branch, so its
frozen pin in ``frozen-hf-dataset-revisions.json`` points at a commit on Hugging Face's auto-generated
``refs/convert/parquet`` branch.
"""

from typing import Any, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_frozen
from eval_framework.tasks.task_style import MCStyle

SOCIAL_I_QA_DATASET_PATH = "allenai/social_i_qa"


@final
class SocialIqaReader(ChoiceReader):
    """Reads a Social IQa item: the shown question is the context followed by the question, with the correct
    option at the 1-indexed ``label``."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(
            raw_question=f"{item['context']} {item['question']}".strip(),
            choices=[item["answerA"], item["answerB"], item["answerC"]],
            correct_index=int(item["label"]) - 1,
        )


def social_iqa_mc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(question_prefix="Question: ", cue_text="Answer:", space_prefixed_labels=True)
    dataset_policy = dataset if dataset is not None else pinned_frozen(SOCIAL_I_QA_DATASET_PATH)
    return ComposedBenchmark.choice(
        id="SocialIQAMC_OLMES",
        reader=SocialIqaReader(),
        styler=styler,
        sample_split="train",
        fewshot_split="train",
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


SOCIAL_IQA_BENCHMARKS: list[Benchmark] = [social_iqa_mc_olmes()]
