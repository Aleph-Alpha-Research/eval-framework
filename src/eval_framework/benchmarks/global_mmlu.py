"""Global-MMLU: https://huggingface.co/datasets/CohereLabs/Global-MMLU

MMLU translated into many languages; we evaluate French, German, Spanish, Italian, Portuguese and Arabic.

Variants:

- ``GlobalMMLU`` / ``GlobalMMLU_German``: multiple choice over all languages / German only, scored over the
  letter labels. The language is part of each subject label, so these use a hand-written kind.
- ``GlobalMMLU_Cohere_{lang}_BPB`` / ``GlobalMMLULite_Cohere_{lang}_BPB``: one task per language (e.g.
  ``GlobalMMLU_Cohere_DE_BPB``), options hidden, BPB of the gold option's full text. Global-MMLU-Lite
  (https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite) is a 400-question subset per language.
"""

import ast
from itertools import product
from typing import TYPE_CHECKING, Any, final, override

from datasets import DatasetDict

from eval_framework.answer import PickFromCandidates
from eval_framework.benchmarks.mmlu import MMLU_SUBJECTS
from eval_framework.benchmarks.utils.global_mmlu_languages import GLOBAL_MMLU_LOCALES
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import EvalKind, SampleBody, assemble_messages
from eval_framework.fewshot import FewShot, FewshotExample, FewShotSplit, FunctionRenderer
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyBayesianLoglikelihood,
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.metrics.loglikelihood.bpb_variants import BitsPerByteVariantsLoglikelihood
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle
from eval_framework.tasks.utils import get_n_letters
from template_formatting.formatter import Message

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric

GLOBAL_MMLU_DATASET_PATH = "CohereLabs/Global-MMLU"
GLOBAL_MMLU_LANGUAGES = ["fr", "de", "es", "it", "pt", "ar", "nl", "pl", "tr", "ru", "uk", "ro"]

GLOBAL_MMLU_LITE_DATASET_PATH = "CohereLabs/Global-MMLU-Lite"
# Our languages that Global-MMLU-Lite also has (it lacks nl, pl, tr, ru, uk and ro).
GLOBAL_MMLU_LITE_LANGUAGES = ["fr", "de", "es", "it", "pt", "ar"]
# Lite's test split covers only 43 of the 57 MMLU subjects (the same ones in every language).
_LITE_MISSING_SUBJECTS = {
    "abstract_algebra",
    "anatomy",
    "clinical_knowledge",
    "college_computer_science",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "formal_logic",
    "high_school_biology",
    "machine_learning",
    "medical_genetics",
    "moral_scenarios",
    "world_religions",
}
GLOBAL_MMLU_LITE_SUBJECTS = [subject for subject in MMLU_SUBJECTS if subject not in _LITE_MISSING_SUBJECTS]

_OPTION_KEYS = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}
_KEYS = get_n_letters(4)  # A, B, C, D


def _lang_and_subject(subject_label: str) -> tuple[str, str]:
    """Parse a ``"('de', 'abstract_algebra')"`` subject label into its (language, english subject) parts."""
    lang, subject = ast.literal_eval(subject_label)
    return lang, subject


def _mc_prompt(item: dict[str, Any], language_key: str) -> str:
    question = item["question"].strip()
    choices = "".join(f"{key}. {item[_OPTION_KEYS[key]]}\n" for key in _KEYS)
    return f"{GLOBAL_MMLU_LOCALES[language_key].question_label}: {question}\n{choices}"


class _GlobalMmluLoader(DatasetLoader):
    """Loads one ``(language, subject)`` slice: the language names the config, the subject filters the rows."""

    def __init__(self, inner: DatasetLoader) -> None:
        self._inner = inner

    @override
    def load(self, name: str | None) -> DatasetDict:
        assert name is not None, "GlobalMMLU subjects always carry a (language, subject) load key."
        lang, subject = _lang_and_subject(name)
        loaded = self._inner.load(lang)
        # Tag each row with its language (implicit in the config, absent from the row) so a demonstration can
        # be rendered from the row alone — the localized "Question:"/"Answer:" labels need it.
        return DatasetDict(
            {
                split: data.filter(lambda row: row["subject"] == subject).map(lambda row: {"language": lang})
                for split, data in loaded.items()
            }
        )

    @override
    def metadata(self) -> dict[str, str]:
        return self._inner.metadata()


@final
class _GlobalMmluDataset(DatasetPolicy):
    """Global-MMLU's data policy: each subject is a ``(language, subject)`` pair — config by language,
    filtered by the ``subject`` column."""

    def __init__(self, inner: DatasetPolicy, dataset_path: str) -> None:
        self._inner = inner
        self._dataset_path = dataset_path

    @override
    def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
        return _GlobalMmluLoader(self._inner.loader(custom_hf_revision))

    @override
    def documentation(self) -> str:
        url = f"https://huggingface.co/datasets/{self._dataset_path}"
        return (
            f"- Link to dataset: [{url}]({url})\n"
            "- Each subject is a `(language, subject)` pair: the language selects the config, and the subject "
            "is kept from the `subject` column."
        )


@final
class _GlobalMmluChoice(EvalKind):
    """Localized multiple-choice loglikelihood: the preamble, the "Question"/"Answer" labels and the subject
    name are rendered in the subject's language (encoded in the subject label); scored over the four letters."""

    @override
    def metrics(self) -> list[type["BaseMetric"]]:
        return [
            AccuracyLoglikelihood,
            AccuracyNormLoglikelihood,
            AccuracyBayesianLoglikelihood,
            BitsPerByteLoglikelihood,
            BitsPerByteVariantsLoglikelihood,
        ]

    @override
    def messages(self, body: SampleBody, *, fewshot: list[FewshotExample], subject_label: str) -> list[Message]:
        lang, subject = _lang_and_subject(subject_label)
        return assemble_messages(fewshot, body, initial_prompt=GLOBAL_MMLU_LOCALES[lang].preamble(subject))

    @override
    def samples(self, item: dict[str, Any]) -> list[SampleBody]:
        lang, _ = _lang_and_subject(item["subject"])
        return [
            SampleBody(
                prompt=_mc_prompt(item, lang),
                cue=f"{GLOBAL_MMLU_LOCALES[lang].answer_label}:",
                possible_completions=[f" {key}" for key in _KEYS],
                ground_truth=f" {item['answer']}",
            )
        ]


def _global_mmlu_demo(row: dict[str, Any]) -> FewshotExample:
    # The demonstration is rendered in the row's own language; the pool is the same (language, subject) slice
    # as the eval item, so this matches the item's language.
    lang = row["language"]
    return FewshotExample(
        prompt=_mc_prompt(row, lang),
        answer=f"{GLOBAL_MMLU_LOCALES[lang].answer_label}: {row['answer']}",
    )


def _global_mmlu_dataset(dataset: DatasetPolicy | None) -> DatasetPolicy:
    return dataset if dataset is not None else _global_mmlu_pinned(GLOBAL_MMLU_DATASET_PATH)


def _global_mmlu_pinned(dataset_path: str) -> DatasetPolicy:
    return _GlobalMmluDataset(pinned_by_framework(dataset_path), dataset_path)


def _global_mmlu(id: str, subjects: ListOfSubjects, dataset: DatasetPolicy | None) -> Benchmark:
    return ComposedBenchmark.compose(
        id=id,
        kind=_GlobalMmluChoice(),
        answer=PickFromCandidates(),
        sample_split="test",
        fewshot=FewShot(FewShotSplit("dev"), FunctionRenderer(_global_mmlu_demo)),
        subjects=subjects,
        dataset_policy=_global_mmlu_dataset(dataset),
    )


def global_mmlu(dataset: DatasetPolicy | None = None) -> Benchmark:
    subjects = ListOfSubjects([str(pair) for pair in product(GLOBAL_MMLU_LANGUAGES, MMLU_SUBJECTS)])
    return _global_mmlu("GlobalMMLU", subjects, dataset)


def global_mmlu_german(dataset: DatasetPolicy | None = None) -> Benchmark:
    subjects = ListOfSubjects([str(("de", subject)) for subject in MMLU_SUBJECTS])
    return _global_mmlu("GlobalMMLU_German", subjects, dataset)


# --- per-language BPB variants ------------------------------------------------------------------------------


@final
class GlobalMmluReader(ChoiceReader):
    """Reads a Global-MMLU (or -Lite) item: the question, its four options and the gold letter as an index."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(
            raw_question=item["question"].strip(),
            choices=[item[_OPTION_KEYS[key]] for key in _KEYS],
            correct_index=_KEYS.index(item["answer"]),
        )


def global_mmlu_bpb(language: str, lite: bool = False, dataset: DatasetPolicy | None = None) -> Benchmark:
    """Build ``GlobalMMLU[Lite]_Cohere_{language}_BPB``, e.g. ``global_mmlu_bpb("de")``: the localized cloze prompt
    (no options, no "multiple choice" in the preamble), scoring only the gold option's full text."""
    locale = GLOBAL_MMLU_LOCALES[language]
    styler = BPBStyle(
        question_prefix=f"{locale.question_label}: ",
        cue_text=f"{locale.answer_label}:",
        initial_prompt=lambda subject_label: locale.cloze_preamble(_lang_and_subject(subject_label)[1]),
    )
    subjects = GLOBAL_MMLU_LITE_SUBJECTS if lite else MMLU_SUBJECTS
    dataset_path = GLOBAL_MMLU_LITE_DATASET_PATH if lite else GLOBAL_MMLU_DATASET_PATH
    return ComposedBenchmark.choice(
        id=f"GlobalMMLU{'Lite' if lite else ''}_Cohere_{language.upper()}_BPB",
        reader=GlobalMmluReader(),
        styler=styler,
        sample_split="test",
        fewshot_split="dev",
        # Same (language, subject) labels as the multiple-choice tasks, so the same loader serves them.
        subjects=ListOfSubjects([str((language, subject)) for subject in subjects]),
        dataset_policy=dataset if dataset is not None else _global_mmlu_pinned(dataset_path),
    )


GLOBAL_MMLU_BENCHMARKS: list[Benchmark] = [
    global_mmlu(),
    global_mmlu_german(),
    *(global_mmlu_bpb(language) for language in GLOBAL_MMLU_LANGUAGES),
    *(global_mmlu_bpb(language, lite=True) for language in GLOBAL_MMLU_LITE_LANGUAGES),
]
