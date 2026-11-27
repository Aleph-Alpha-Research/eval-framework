"""CommonsenseQA (English): https://huggingface.co/datasets/tau/commonsense_qa

Five-way multiple-choice commonsense questions. The variants:

* ``CSQA_Tau_EN_MC_OLMES`` — the options are shown as space-prefixed lettered choices (" A. …") and the
  model is scored over the letter labels. Mirrors OLMES' OLMo 3 Base Main task (``csqa:mc::xlarge``).
"""

from typing import Any, final, override

from eval_framework.answer import PickFromCandidates
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice
from eval_framework.fewshot import ChoiceRenderer, FewShot, FirstOf
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import MCStyle, TaskStyler

_DATASET_PATH = "tau/commonsense_qa"

# OLMES' fixed demonstrations for CommonsenseQA (its "OLMES:commonsense_qa" few-shot source)
_OLMES_FEWSHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "question": "Sammy wanted to go to where the people were.  Where might he go?",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": ["race track", "populated areas", "the desert", "apartment", "roadblock"],
        },
        "answerKey": "B",
    },
    {
        "question": "Google Maps and other highway and street GPS services have replaced what?",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": ["united states", "mexico", "countryside", "atlas", "oceans"],
        },
        "answerKey": "D",
    },
    {
        "question": "The fox walked from the city into the forest, what was it looking for?",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": ["pretty flowers.", "hen house", "natural habitat", "storybook", "dense forest"],
        },
        "answerKey": "C",
    },
    {
        "question": "What is it called when you slowly cook using a grill?",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": ["backyard", "restaurant", "crockpot", "neighbor's house", "barbeque"],
        },
        "answerKey": "E",
    },
    {
        "question": "What would you do if you want to be able to earn money?",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": ["apply for job", "stand in line", "take care of proposals", "pass course", "play the lottery"],
        },
        "answerKey": "A",
    },
    {
        "question": "What home entertainment equipment requires cable?",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": ["radio shack", "substation", "cabinet", "television", "desk"],
        },
        "answerKey": "D",
    },
    {
        "question": "Of all the rooms in a house it was his favorite, the aromas always drew him to the what?",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": ["yard", "basement", "kitchen", "living room", "garden"],
        },
        "answerKey": "C",
    },
]


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


def _dataset_policy(dataset: DatasetPolicy | None) -> DatasetPolicy:
    return dataset if dataset is not None else pinned_by_framework(_DATASET_PATH)


def _csqa_olmes(id: str, styler: TaskStyler, sample_split: str, dataset: DatasetPolicy | None) -> Benchmark:
    """A CommonsenseQA task showing the first ``num_fewshot`` of OLMES' fixed demonstrations."""
    reader = CommonsenseQaReader()
    return ComposedBenchmark.compose(
        id=id,
        kind=Choice(reader, styler),
        answer=PickFromCandidates(),
        sample_split=sample_split,
        fewshot=FewShot(FirstOf(_OLMES_FEWSHOT_EXAMPLES, label=id), ChoiceRenderer(reader, styler)),
        dataset_policy=_dataset_policy(dataset),
    )


def csqa_mc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    """Mostly matches OLMo 3 Base Main, OLMES `csqa:mc::xlarge`.

    They use a random 10,000 subset of train + validation. We use the largest single split `train` as the
    approximation.
    """
    styler = MCStyle(question_prefix="Question: ", cue_text="Answer:", space_prefixed_labels=True)
    return _csqa_olmes("CSQA_Tau_EN_MC_OLMES", styler, "train", dataset)


CSQA_BENCHMARKS: list[Benchmark] = [csqa_mc_olmes()]
