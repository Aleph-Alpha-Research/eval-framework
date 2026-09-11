"""MMLU: https://huggingface.co/datasets/cais/mmlu

Multiple-choice knowledge questions across 57 subjects (each an HF config, loaded per subject); every
prompt is prefaced by a subject-templated preamble. The composed variants:

* ``MMLU`` / ``MMLU_OLMES`` — score the letter labels; OLMES adds a space before each option label.
* ``Full Text MMLU`` — shows the options as a bulleted list and scores the full answer text.
* ``MMLU_IDK`` — lets the model abstain with ``"?"`` and reports confidence-aware metrics.

``MMLU_COT`` stays BaseTask until the composed design supports generative (completion) tasks.
"""

from typing import Any, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.benchmarks.mmlu import MMLU_SUBJECTS  # single source of truth for the 57 subjects
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import ClozeStyle, MCStyle, TaskStyler


@final
class MmluReader(ChoiceReader):
    """Reads an MMLU item: the question and its four answer choices, with the correct one at ``answer``."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(
            raw_question=item["question"].strip(),
            choices=item["choices"],
            correct_index=item["answer"],
        )


@final
class _FullTextMmluStyle(ClozeStyle):
    """Cloze scoring (full answer text) that additionally shows the options as a bulleted list."""

    @override
    def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
        options = "".join(f"- {choice}\n" for choice in choices)
        return f"{self.question_prefix}{raw_question}\nPossible answers:\n{options}"


def _humanized(subject_label: str) -> str:
    # Subject labels are underscored config names ("abstract_algebra"); read them as prose.
    return " ".join(subject_label.split("_"))


def _mc_preamble(subject_label: str) -> str:
    return f"The following are multiple choice questions (with answers) about {_humanized(subject_label)}."


def _full_text_preamble(subject_label: str) -> str:
    return (
        f"The following are multiple choice questions (with possible answers) about {_humanized(subject_label)}.\n"
        "Answer with the full text of the correct answer."
    )


def _idk_preamble(subject_label: str) -> str:
    # Unlike the other variants, the IDK preamble names the subject by its raw (underscored) key.
    return (
        f"The following are multiple choice questions (with answers) about {subject_label}. "
        "Answer only if you are confident, since mistakes may be penalised, while correct answers receive points. "
        "It is acceptable to answer with '?' if you are unsure, and you will receive 0 points."
    )


def _mmlu_benchmark(
    id: str, styler: TaskStyler, dataset: DatasetPolicy | None = None, display_name: str | None = None
) -> Benchmark:
    dataset_policy = dataset if dataset is not None else pinned_by_framework("cais/mmlu")
    return ComposedBenchmark.compose(
        id=id,
        display_name=display_name,
        kind=Choice(reader=MmluReader(), styler=styler),
        sample_split="test",
        fewshot_split="dev",
        subjects=ListOfSubjects(MMLU_SUBJECTS),
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


def mmlu(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(question_prefix="Question: ", cue_text="Answer:", initial_prompt=_mc_preamble)
    return _mmlu_benchmark("MMLU", styler, dataset)


def mmlu_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(
        question_prefix="Question: ", cue_text="Answer:", space_prefixed_labels=True, initial_prompt=_mc_preamble
    )
    return _mmlu_benchmark("MMLU_OLMES", styler, dataset)


def mmlu_full_text(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = _FullTextMmluStyle(question_prefix="Question: ", cue_text="Answer:", initial_prompt=_full_text_preamble)
    # The registry/hash identity is the compact "FullTextMMLU"; the display name keeps the spelled-out form.
    return _mmlu_benchmark("FullTextMMLU", styler, dataset, display_name="Full Text MMLU")


def mmlu_idk(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(
        question_prefix="Question: ", cue_text="Answer:", initial_prompt=_idk_preamble
    ).with_abstention_option(" ?")
    return _mmlu_benchmark("MMLU_IDK", styler, dataset)


MMLU_BENCHMARKS: list[Benchmark] = [mmlu(), mmlu_olmes(), mmlu_full_text(), mmlu_idk()]
