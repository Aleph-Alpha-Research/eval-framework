"""Global-MMLU: https://huggingface.co/datasets/CohereLabs/Global-MMLU

MMLU translated into many languages; we evaluate French, German, Spanish, Italian, Portuguese and Arabic.
"""

import ast
from itertools import product
from typing import TYPE_CHECKING, Any, final, override

from datasets import DatasetDict

from eval_framework.answer import PickFromCandidates
from eval_framework.benchmarks.mmlu import MMLU_SUBJECTS
from eval_framework.benchmarks.utils.global_mmlu_languages import GLOBAL_MMLU_LOCALES
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
from eval_framework.tasks.utils import get_n_letters
from template_formatting.formatter import Message

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric

GLOBAL_MMLU_DATASET_PATH = "CohereLabs/Global-MMLU"
GLOBAL_MMLU_LANGUAGES = ["fr", "de", "es", "it", "pt", "ar"]

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

    def __init__(self, inner: DatasetPolicy) -> None:
        self._inner = inner

    @override
    def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
        return _GlobalMmluLoader(self._inner.loader(custom_hf_revision))

    @override
    def documentation(self) -> str:
        url = f"https://huggingface.co/datasets/{GLOBAL_MMLU_DATASET_PATH}"
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
    return dataset if dataset is not None else _GlobalMmluDataset(pinned_by_framework(GLOBAL_MMLU_DATASET_PATH))


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


GLOBAL_MMLU_BENCHMARKS: list[Benchmark] = [global_mmlu(), global_mmlu_german()]
