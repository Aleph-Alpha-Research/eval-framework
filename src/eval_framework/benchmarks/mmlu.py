"""MMLU: https://huggingface.co/datasets/cais/mmlu

Multiple-choice knowledge questions across 57 subjects (each an HF config, loaded per subject); every
prompt is prefaced by a subject-templated preamble. The composed variants:

* ``MMLU`` / ``MMLU_OLMES`` — score the letter labels; OLMES adds a space before each option label.
* ``Full Text MMLU`` — shows the options as a bulleted list and scores the full answer text.
* ``MMLU_IDK`` — lets the model abstain with ``"?"`` and reports confidence-aware metrics.
* ``MMLU_COT`` — the model reasons freely and concludes with the answer, which is regex-extracted from
  the generation (free-form completion, 0-shot).
"""

import re
from typing import Any, final, override

from eval_framework.answer import ExtractFromCompletion, first_match
from eval_framework.benchmarks.cot import Cot
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.fewshot import NoFewShot
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import ClozeStyle, MCStyle, TaskStyler
from eval_framework.tasks.utils import get_n_letters

# The 57 MMLU subjects; each is an HF config of ``cais/mmlu`` (also reused by GlobalMMLU).
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


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


def _mmlu_cot_preamble(subject_label: str) -> str:
    return (
        f"The following are multiple choice questions about {_humanized(subject_label)}. "
        'Summarize your reasoning concisely, then conclude with "Therefore, the answer is: X", where X is '
        "one of A, B, C, or D."
    )


def _mmlu_cot_prompt(raw_question: str, choices: list[str]) -> str:
    keys = get_n_letters(len(choices))
    options = "\n".join(f"{key}. {choice}" for key, choice in zip(keys, choices))
    return f"Question: {raw_question}\n{options}"


_MMLU_COT_ANSWER_RE = re.compile(r"Therefore, the answer is: ([ABCD])")


def _mmlu_dataset(dataset: DatasetPolicy | None) -> DatasetPolicy:
    return dataset if dataset is not None else pinned_by_framework("cais/mmlu")


def _mmlu_choice(
    id: str, styler: TaskStyler, dataset: DatasetPolicy | None = None, display_name: str | None = None
) -> Benchmark:
    # The loglikelihood variants differ only in their styler; MmluReader and the dev fewshot split are shared.
    return ComposedBenchmark.choice(
        id=id,
        display_name=display_name,
        reader=MmluReader(),
        styler=styler,
        sample_split="test",
        fewshot_split="dev",
        subjects=ListOfSubjects(MMLU_SUBJECTS),
        dataset_policy=_mmlu_dataset(dataset),
        language=Language.ENG,
    )


def mmlu(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(question_prefix="Question: ", cue_text="Answer:", initial_prompt=_mc_preamble)
    return _mmlu_choice("MMLU", styler, dataset)


def mmlu_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(
        question_prefix="Question: ", cue_text="Answer:", space_prefixed_labels=True, initial_prompt=_mc_preamble
    )
    return _mmlu_choice("MMLU_OLMES", styler, dataset)


def mmlu_full_text(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = _FullTextMmluStyle(question_prefix="Question: ", cue_text="Answer:", initial_prompt=_full_text_preamble)
    # The registry/hash identity is the compact "FullTextMMLU"; the display name keeps the spelled-out form.
    return _mmlu_choice("FullTextMMLU", styler, dataset, display_name="Full Text MMLU")


def mmlu_idk(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(
        question_prefix="Question: ", cue_text="Answer:", initial_prompt=_idk_preamble
    ).with_abstention_option(" ?")
    return _mmlu_choice("MMLU_IDK", styler, dataset)


def mmlu_cot(dataset: DatasetPolicy | None = None) -> Benchmark:
    # Free-form (completion) and 0-shot only, so it takes the general compose path with the shared Cot kind.
    return ComposedBenchmark.compose(
        id="MMLU_COT",
        kind=Cot(MmluReader(), build_prompt=_mmlu_cot_prompt, preamble=_mmlu_cot_preamble),
        answer=ExtractFromCompletion(first_match(_MMLU_COT_ANSWER_RE), ["Question:"]),
        sample_split="test",
        fewshot=NoFewShot(),
        subjects=ListOfSubjects(MMLU_SUBJECTS),
        dataset_policy=_mmlu_dataset(dataset),
        language=Language.ENG,
    )


MMLU_BENCHMARKS: list[Benchmark] = [mmlu(), mmlu_olmes(), mmlu_full_text(), mmlu_idk(), mmlu_cot()]
