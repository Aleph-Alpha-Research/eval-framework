"""COPA (Choice of Plausible Alternatives): https://huggingface.co/datasets/aps/super_glue

Causal-reasoning items: a premise, a cause/effect cue, and two alternatives. The registered variant is
OLMES-style: the premise is recast as a sentence stem (its final period replaced by the causal connector),
the two alternatives are shown as space-prefixed lettered options (" A. …"), and the model is scored over
the letter labels. The prompt carries no assistant cue — the options directly continue the stem.
"""

from typing import Any, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import MCStyle

# The premise becomes a sentence stem: its trailing period is replaced by the causal connector, so each
# alternative continues it (e.g. "The man broke his toe because" + " he dropped a hammer on it.").
_COPA_CONNECTOR = {"cause": "because", "effect": "therefore"}


def _decapitalize(text: str) -> str:
    return text[0].lower() + text[1:]


@final
class CopaReader(ChoiceReader):
    """Reads a COPA item: the premise recast as a stem (final period → connector) and its two alternatives,
    each lower-cased to continue the stem."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        premise = item["premise"].strip()[:-1] + f" {_COPA_CONNECTOR[item['question']]}"
        return ChoiceFields(
            raw_question=premise,
            choices=[_decapitalize(item["choice1"]), _decapitalize(item["choice2"])],
            correct_index=item["label"],
        )


def copa_mc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(question_prefix="", cue_text="", space_prefixed_labels=True)
    dataset_policy = dataset if dataset is not None else pinned_by_framework("aps/super_glue")
    return ComposedBenchmark.choice(
        id="COPA_OLMES",
        reader=CopaReader(),
        styler=styler,
        sample_split="validation",
        fewshot_split="test",
        subjects=ListOfSubjects(["copa"]),
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


COPA_BENCHMARKS: list[Benchmark] = [copa_mc_olmes()]
