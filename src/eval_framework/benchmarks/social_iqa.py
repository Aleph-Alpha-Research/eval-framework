"""Social IQa (English): https://huggingface.co/datasets/allenai/social_i_qa

Commonsense reasoning about social situations: a context and question with three answer options
(``answerA``/``answerB``/``answerC``) and a 1-indexed ``label`` marking the correct one. The variants:

* ``SIQA_AllenAI_EN_MC_OLMES`` — the options are shown as space-prefixed lettered choices (" A. …") and the
  model is scored over the letter labels. Mirrors OLMES' OLMo 3 Base Main task (``socialiqa:mc::xlarge``),
  showing the first ``num_fewshot`` of OLMES' fixed demonstrations.

``allenai/social_i_qa`` ships only a (no-longer-supported) loading script on its main branch, so its
frozen pin in ``frozen-hf-dataset-revisions.json`` points at a commit on Hugging Face's auto-generated
``refs/convert/parquet`` branch.
"""

from typing import Any, final, override

from eval_framework.answer import PickFromCandidates
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice
from eval_framework.fewshot import ChoiceRenderer, FewShot, FewShotSource, FirstOf
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_frozen
from eval_framework.tasks.task_style import MCStyle, TaskStyler

SOCIAL_I_QA_DATASET_PATH = "allenai/social_i_qa"

# OLMES' fixed demonstrations for Social IQa (its "OLMES:social_i_qa" few-shot source)
_OLMES_FEWSHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "context": "Cameron decided to have a barbecue and gathered her friends together.",
        "question": "How would Others feel as a result?",
        "answerA": "like attending",
        "answerB": "like staying home",
        "answerC": "a good friend to have",
        "label": "1",
    },
    {
        "context": "Quinn wanted to help me clean my room up because it was so messy.",
        "question": "What will Quinn want to do next?",
        "answerA": "Eat messy snacks",
        "answerB": "help out a friend",
        "answerC": "Pick up the dirty clothes",
        "label": "3",
    },
    {
        "context": "Jan needed to give out jobs for an upcoming project at work.",
        "question": "What will Others want to do next?",
        "answerA": "disagree with Jan",
        "answerB": "get to work",
        "answerC": "argue with the assignments",
        "label": "2",
    },
    {
        "context": "Their cat kept trying to escape out of the window, so Jan placed an obstacle in the way.",
        "question": "How would Jan feel afterwards?",
        "answerA": "scared of losing the cat",
        "answerB": "normal",
        "answerC": "relieved for fixing the problem",
        "label": "3",
    },
    {
        "context": "Remy was an expert fisherman and was on the water with Kai. Remy baited Kai's hook.",
        "question": "What will Remy want to do next?",
        "answerA": "cast the line",
        "answerB": "put the boat in the water",
        "answerC": "invite Kai out on the boat",
        "label": "1",
    },
    {
        "context": "Kendall worked the weekend at the steakhouse and made bank on tips.",
        "question": "What will Kendall want to do next?",
        "answerA": "Save the money",
        "answerB": "get hired at the steakhouse",
        "answerC": "Quit her job",
        "label": "1",
    },
    {
        "context": "Bailey relieved every one of her friends when she announced her plans to stay.",
        "question": "Why did Bailey do this?",
        "answerA": "wanted to live by herself",
        "answerB": "wanted to show her dedication to her friends",
        "answerC": "wanted to forget about her friends",
        "label": "2",
    },
    {
        "context": "Kendall ran back and thanked Lee for helping her find the dog.",
        "question": "How would you describe Kendall?",
        "answerA": "grateful",
        "answerB": "super",
        "answerC": "amazing",
        "label": "1",
    },
]


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


def _dataset_policy(dataset: DatasetPolicy | None) -> DatasetPolicy:
    return dataset if dataset is not None else pinned_frozen(SOCIAL_I_QA_DATASET_PATH)


def _social_iqa(
    id: str, styler: TaskStyler, *, sample_split: str, fewshot_source: FewShotSource, dataset: DatasetPolicy | None
) -> Benchmark:
    reader = SocialIqaReader()
    return ComposedBenchmark.compose(
        id=id,
        kind=Choice(reader, styler),
        answer=PickFromCandidates(),
        sample_split=sample_split,
        fewshot=FewShot(fewshot_source, ChoiceRenderer(reader, styler)),
        dataset_policy=_dataset_policy(dataset),
    )


# The OLMES variants: the first `num_fewshot` of OLMES' fixed demonstrations.
_OLMES_DEMONSTRATIONS = FirstOf(_OLMES_FEWSHOT_EXAMPLES, label="Social IQa (OLMES demonstrations)")


def social_iqa_mc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    """Mostly matches OLMo 3 Base Main, OLMES `socialiqa:mc::xlarge`.

    They use a random 10,000 subset of train + validation. We use the largest single split `train` as the
    approximation.
    """
    return _social_iqa(
        "SIQA_AllenAI_EN_MC_OLMES",
        MCStyle(space_prefixed_labels=True),
        sample_split="train",
        fewshot_source=_OLMES_DEMONSTRATIONS,
        dataset=dataset,
    )


SOCIAL_IQA_BENCHMARKS: list[Benchmark] = [social_iqa_mc_olmes()]
