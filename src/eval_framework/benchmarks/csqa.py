"""CommonsenseQA (English): https://huggingface.co/datasets/tau/commonsense_qa

Five-way multiple-choice commonsense questions. The variants:

* ``CSQA_Tau_EN_MC`` — the options are shown as lettered choices ("A. …") and the model is scored over the
  letter labels.
* ``CSQA_Tau_EN_Cloze`` — the options are hidden and the model is scored over the full answer texts.
* ``CSQA_Tau_EN_BPB`` — the cloze prompt, but only the BPB of the gold answer's full text is scored.

These evaluate the validation split with demonstrations drawn at random from train. The ``_OLMES`` variants
instead follow OLMES' OLMo 3 suites, showing the first ``num_fewshot`` of OLMES' fixed demonstrations:

* ``CSQA_Tau_EN_MC_OLMES`` — the options are shown as space-prefixed lettered choices (" A. …").
  Mirrors OLMo 3 Base Main (``csqa:mc::xlarge``).
* ``CSQA_Tau_EN_Cloze_OLMES``, ``CSQA_Tau_EN_BPB_OLMES`` — mirror OLMo 3 Base Easy
  (``csqa:rc::olmes:full``, ``csqa:rc:bpb::olmes:full``).
"""

from typing import Any, final, override

from eval_framework.answer import PickFromCandidates
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice
from eval_framework.fewshot import ChoiceRenderer, FewShot, FewShotSource, FewShotSplit, FirstOf
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, TaskStyler

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


def _csqa(
    id: str, styler: TaskStyler, *, sample_split: str, fewshot_source: FewShotSource, dataset: DatasetPolicy | None
) -> Benchmark:
    reader = CommonsenseQaReader()
    return ComposedBenchmark.compose(
        id=id,
        kind=Choice(reader, styler),
        answer=PickFromCandidates(),
        sample_split=sample_split,
        fewshot=FewShot(fewshot_source, ChoiceRenderer(reader, styler)),
        dataset_policy=_dataset_policy(dataset),
    )


# The generic variants: the validation split, with demonstrations drawn at random from train.
_FROM_TRAIN = FewShotSplit("train")

# The OLMES variants: the first `num_fewshot` of OLMES' fixed demonstrations.
_OLMES_DEMONSTRATIONS = FirstOf(_OLMES_FEWSHOT_EXAMPLES, label="CommonsenseQA (OLMES demonstrations)")


def csqa_mc(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _csqa(
        "CSQA_Tau_EN_MC",
        MCStyle(),
        sample_split="validation",
        fewshot_source=_FROM_TRAIN,
        dataset=dataset,
    )


def csqa_mc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    """Mostly matches OLMo 3 Base Main, OLMES `csqa:mc::xlarge`.

    They use a random 10,000 subset of train + validation. We use the largest single split `train` as the
    approximation.
    """
    return _csqa(
        "CSQA_Tau_EN_MC_OLMES",
        MCStyle(space_prefixed_labels=True),
        sample_split="train",
        fewshot_source=_OLMES_DEMONSTRATIONS,
        dataset=dataset,
    )


def csqa_cloze(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _csqa(
        "CSQA_Tau_EN_Cloze",
        ClozeStyle(),
        sample_split="validation",
        fewshot_source=_FROM_TRAIN,
        dataset=dataset,
    )


def csqa_cloze_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    """Matches OLMo 3 Base Easy, OLMES `csqa:rc::olmes:full`, which uses the full validation split."""
    return _csqa(
        "CSQA_Tau_EN_Cloze_OLMES",
        ClozeStyle(),
        sample_split="validation",
        fewshot_source=_OLMES_DEMONSTRATIONS,
        dataset=dataset,
    )


def csqa_bpb(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _csqa(
        "CSQA_Tau_EN_BPB",
        BPBStyle(),
        sample_split="validation",
        fewshot_source=_FROM_TRAIN,
        dataset=dataset,
    )


def csqa_bpb_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    """Matches OLMo 3 Base Easy, OLMES `csqa:rc:bpb::olmes:full`, which uses the full validation split."""
    return _csqa(
        "CSQA_Tau_EN_BPB_OLMES",
        BPBStyle(),
        sample_split="validation",
        fewshot_source=_OLMES_DEMONSTRATIONS,
        dataset=dataset,
    )


CSQA_BENCHMARKS: list[Benchmark] = [
    csqa_mc(),
    csqa_mc_olmes(),
    csqa_cloze(),
    csqa_cloze_olmes(),
    csqa_bpb(),
    csqa_bpb_olmes(),
]
