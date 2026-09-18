"""MMLU-Pro: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

Harder, ten-option multiple-choice questions across 14 categories. All questions live in one config and are
split into subjects by the ``category`` column. Every prompt is prefaced by a subject-templated preamble.
The composed variants:

Note: MMLU-Pro questions carry a *variable* number of options (6–10), but every loglikelihood variant scores
a fixed ten letters A–J regardless (see ``_MmluProMCStyle``) — A bug faithfully preserved from the original
implementation, in order to not change the meaning of the score silently.
"""

from typing import Any, final, override

from eval_framework.answer import AnswerPolicy
from eval_framework.benchmarks.cot import Cot, tulu3_cot_prompt, tulu_answer, tulu_answer_v2
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.fewshot import NoFewShot
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import MCStyle, TaskStyler
from eval_framework.tasks.utils import get_n_letters

MMLU_PRO_SUBJECTS = [
    "engineering",
    "physics",
    "psychology",
    "chemistry",
    "biology",
    "law",
    "philosophy",
    "computer science",
    "other",
    "economics",
    "business",
    "history",
    "math",
    "health",
]


@final
class MmluProReader(ChoiceReader):
    """Reads an MMLU-Pro item: the question and its (6–10) options, with the correct one at ``answer_index``."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(
            raw_question=item["question"].strip(),
            choices=item["options"],
            correct_index=item["answer_index"],
        )


@final
class _MmluProMCStyle(MCStyle):
    """Multiple-choice styler that scores a fixed ten letters (A–J) regardless of how many options a question
    actually lists. MMLU-Pro questions have a variable option count, but every loglikelihood variant scores
    all ten letters — a quirk preserved from the original task (letters past the real options can never be the
    answer, so they can only ever cost accuracy)."""

    @override
    def get_possible_completions(self, choices: list[str], correct_index: int | None = None) -> list[str]:
        return [f" {label}" for label in get_n_letters(10)]


def _mmlu_pro_preamble(subject_label: str) -> str:
    return f"The following are multiple choice questions (with answers) about {subject_label}."


def _mmlu_pro_idk_preamble(subject_label: str) -> str:
    return (
        f"The following are multiple choice questions (with answers) about {subject_label}. "
        "Answer only if you are confident, since mistakes may be penalised, while correct answers receive points. "
        "It is acceptable to answer with '?' if you are unsure, and you will receive 0 points."
    )


def _mmlu_pro_cot_candidates(keys: list[str]) -> list[str]:
    # Faithful quirk: every loglikelihood variant scores a fixed ten letters A–J, so the COT sample carries
    # them too (inert for free-form scoring). See the module docstring.
    return [f" {label}" for label in get_n_letters(10)]


def _mmlu_pro_dataset(dataset: DatasetPolicy | None) -> DatasetPolicy:
    # The subjects are the rows of the single (default) config, split by the ``category`` column.
    return (
        dataset
        if dataset is not None
        else pinned_by_framework("TIGER-Lab/MMLU-Pro").subject_encoded_in_column(config=None, column="category")
    )


def _mmlu_pro_choice(
    id: str, styler: TaskStyler, dataset: DatasetPolicy | None = None, display_name: str | None = None
) -> Benchmark:
    return ComposedBenchmark.choice(
        id=id,
        display_name=display_name,
        reader=MmluProReader(),
        styler=styler,
        sample_split="test",
        fewshot_split="test",
        subjects=ListOfSubjects(MMLU_PRO_SUBJECTS),
        dataset_policy=_mmlu_pro_dataset(dataset),
        language=Language.ENG,
    )


def mmlu_pro(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = _MmluProMCStyle(question_prefix="", cue_text="Answer:", initial_prompt=_mmlu_pro_preamble)
    return _mmlu_pro_choice("MMLU_PRO", styler, dataset, display_name="MMLU Pro")


def mmlu_pro_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = _MmluProMCStyle(
        question_prefix="", cue_text="Answer:", space_prefixed_labels=True, initial_prompt=_mmlu_pro_preamble
    )
    return _mmlu_pro_choice("MMLU_PRO_OLMES", styler, dataset, display_name="MMLU Pro_OLMES")


def mmlu_pro_idk(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = _MmluProMCStyle(
        question_prefix="", cue_text="Answer:", initial_prompt=_mmlu_pro_idk_preamble
    ).with_abstention_option(" ?")
    return _mmlu_pro_choice("MMLU_PRO_IDK", styler, dataset, display_name="MMLU Pro_IDK")


def _mmlu_pro_cot(id: str, answer: AnswerPolicy, dataset: DatasetPolicy | None = None) -> Benchmark:
    return ComposedBenchmark.compose(
        id=id,
        kind=Cot(
            MmluProReader(),
            build_prompt=tulu3_cot_prompt,
            preamble=_mmlu_pro_preamble,
            candidates=_mmlu_pro_cot_candidates,
        ),
        answer=answer,
        sample_split="test",
        fewshot=NoFewShot(),
        subjects=ListOfSubjects(MMLU_PRO_SUBJECTS),
        dataset_policy=_mmlu_pro_dataset(dataset),
        language=Language.ENG,
    )


def mmlu_pro_cot(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _mmlu_pro_cot("MMLU_PRO_COT", tulu_answer(), dataset)


def mmlu_pro_cot_v2(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _mmlu_pro_cot("MMLU_PRO_COT_V2", tulu_answer_v2(10), dataset)


MMLU_PRO_BENCHMARKS: list[Benchmark] = [
    mmlu_pro(),
    mmlu_pro_olmes(),
    mmlu_pro_idk(),
    mmlu_pro_cot(),
    mmlu_pro_cot_v2(),
]
