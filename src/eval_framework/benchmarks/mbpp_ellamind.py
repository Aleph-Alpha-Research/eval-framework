"""Multilingual MBPP (EllaMind): https://huggingface.co/datasets/ellamind/mbpp-multilingual

The multilingual counterpart of MBPP — the dataset mirrors the ``text`` / ``code`` / ``test_list`` schema, so these
reuse the builders and reconstruct functions from the ``mbpp`` module with a per-language instruction wrapper.
Unlike the English variants (fixed English 3-shot block), these sample demonstrations from the eval split of the
language. Each language is one config of the dataset (``deu``).

Variants, per language:

- ``OLMES`` / ``BPB_OLMES``: the OLMES instruction; generation scored by execution, or BPB of the gold solution.
- ``EvalPlus`` / ``BPB_EvalPlus``: the EvalPlus instruction (normalised line endings); likewise.
- ``EvalPlusInstruct``: the EvalPlus instruction without a forced cue; the code block is extracted from the response.

Adding a language: extend ``LanguageCode`` and add its wording to ``_LOCALES``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_args

from eval_framework.answer import ReconstructProgram
from eval_framework.benchmarks.mbpp import (
    END,
    OLMES_STOPS,
    SingleGoldReader,
    bpb,
    evalplus_reconstruct,
    execution,
    instruct_reconstruct,
    olmes_reconstruct,
)
from eval_framework.benchmarks.utils.ellamind_languages import ellamind_language
from eval_framework.contract import Benchmark
from eval_framework.fewshot import ChoiceRenderer, FewShot, SampleSplit
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle

MBPP_ELLAMIND_DATASET_PATH = "ellamind/mbpp-multilingual"

TaskVariant = Literal["OLMES", "BPB_OLMES", "EvalPlus", "BPB_EvalPlus", "EvalPlusInstruct"]

# The languages this file offers (see ``ellamind_languages`` for the token -> language/config table).
LanguageCode = Literal["DE"]
_LANGUAGES: tuple[LanguageCode, ...] = get_args(LanguageCode)

# Variant -> task-id template. The ids predate the naming convention and are kept for continuity.
_ID_TEMPLATES: dict[TaskVariant, str] = {
    "OLMES": "MBPP{language}_OLMES",
    "BPB_OLMES": "MBPP{language}_BPB_OLMES",
    "EvalPlus": "MBPP{language}_EvalPlus",
    "BPB_EvalPlus": "MBPP{language}_BPB_EvalPlus",
    "EvalPlusInstruct": "MBPP{language}EvalPlusInstruct",
}


@dataclass(frozen=True, kw_only=True)
class MbppLocale:
    """One language's wording of the MBPP instruction wrapper and answer cues."""

    preamble: str
    """Opens the instruction, before the fenced problem text and first test."""
    preamble_stop: str
    """The opening words of ``preamble``, as a stop sequence: an EvalPlus generation that starts a new problem
    is cut there."""
    olmes_cue: str
    """Assistant cue of the OLMES variant; opens the code fence the model continues."""
    evalplus_cue: str
    """Assistant cue of the EvalPlus variants; opens the code fence the model continues."""

    def __post_init__(self) -> None:
        if not self.preamble.startswith(self.preamble_stop.lstrip("\n")):
            raise ValueError(f"preamble_stop {self.preamble_stop!r} must open the preamble {self.preamble!r}")


_LOCALES: dict[LanguageCode, MbppLocale] = {
    "DE": MbppLocale(
        preamble=(
            "Bitte erstelle ein in sich geschlossenes Python-Skript, "
            "das das folgende Problem in einem Markdown-Code-Block löst:"
        ),
        preamble_stop="\nBitte erstelle",
        olmes_cue="Hier ist die fertige Funktion:\n\n```python\n",
        evalplus_cue=(
            "Hier ist ein Python-Skript mit einer in sich geschlossenen Funktion, "
            "die das Problem löst und die entsprechenden Tests besteht:\n```python"
        ),
    ),
}


def _text(item: dict[str, Any]) -> str:
    return item["text"] if "text" in item else item["prompt"]


def _norm_text(item: dict[str, Any]) -> str:
    # EvalPlus additionally normalises Windows line endings in the problem text.
    return _text(item).replace("\r\n", "\n").strip()


def _norm_code(item: dict[str, Any]) -> str:
    return (item["code"] or "").replace("\r\n", "\n").strip()


class _MbppInstruction:
    """Renders the instruction wrapper of one language around an item's problem text and first test."""

    def __init__(self, locale: MbppLocale) -> None:
        self._preamble = locale.preamble

    def olmes(self, item: dict[str, Any]) -> str:
        return f"{self._preamble}\n```\n{_text(item).strip()}\n{item['test_list'][0]}\n```\n"

    def evalplus(self, item: dict[str, Any]) -> str:
        return f"{self._preamble}\n```\n{_norm_text(item)}\n{item['test_list'][0]}\n```\n"


# A variant builder: ``(id, locale, subjects, language, dataset) -> Benchmark``.
_Builder = Callable[[str, MbppLocale, ListOfSubjects, Language, DatasetPolicy], Benchmark]


def _olmes(id: str, locale: MbppLocale, subjects: ListOfSubjects, lang: Language, dataset: DatasetPolicy) -> Benchmark:
    return execution(
        id,
        dataset_path=MBPP_ELLAMIND_DATASET_PATH,
        instruction=_MbppInstruction(locale).olmes,
        cue=locale.olmes_cue,
        fewshot_target=lambda demo: demo["code"] + "\n",
        answer=ReconstructProgram(olmes_reconstruct, stop_sequences=OLMES_STOPS),
        fewshot_source=SampleSplit(),  # EllaMind ships only a test split; sample (leak-safe) from it
        subjects=subjects,
        language=lang,
        dataset=dataset,
    )


def _bpb_olmes(
    id: str, locale: MbppLocale, subjects: ListOfSubjects, lang: Language, dataset: DatasetPolicy
) -> Benchmark:
    # The BPB_OLMES prompt carries no instruction wrapper, so nothing of the locale is rendered.
    reader = SingleGoldReader(
        question=lambda item: _norm_text(item) + "\n```python\n",
        gold=lambda item: _norm_code(item) + "\n```",
    )
    styler = BPBStyle(question_prefix="", cue_text="", trailing_newline=False, leading_space_continuations=False)
    return bpb(
        id,
        dataset_path=MBPP_ELLAMIND_DATASET_PATH,
        reader=reader,
        styler=styler,
        fewshot=FewShot(SampleSplit(), ChoiceRenderer(reader, styler)),
        subjects=subjects,
        language=lang,
        dataset=dataset,
    )


def _evalplus(
    id: str, locale: MbppLocale, subjects: ListOfSubjects, lang: Language, dataset: DatasetPolicy
) -> Benchmark:
    return execution(
        id,
        dataset_path=MBPP_ELLAMIND_DATASET_PATH,
        instruction=_MbppInstruction(locale).evalplus,
        cue=locale.evalplus_cue,
        fewshot_target=lambda demo: locale.evalplus_cue + "\n" + _norm_code(demo) + "\n```",
        answer=ReconstructProgram(evalplus_reconstruct, stop_sequences=["```", "\nassert", locale.preamble_stop]),
        fewshot_source=SampleSplit(),
        subjects=subjects,
        language=lang,
        dataset=dataset,
    )


def _bpb_evalplus(
    id: str, locale: MbppLocale, subjects: ListOfSubjects, lang: Language, dataset: DatasetPolicy
) -> Benchmark:
    reader = SingleGoldReader(
        question=_MbppInstruction(locale).evalplus,
        gold=lambda item: "\n" + _norm_code(item) + "\n" + END,
    )
    styler = BPBStyle(
        question_prefix="", cue_text=locale.evalplus_cue, trailing_newline=False, leading_space_continuations=False
    )
    return bpb(
        id,
        dataset_path=MBPP_ELLAMIND_DATASET_PATH,
        reader=reader,
        styler=styler,
        fewshot=FewShot(SampleSplit(), ChoiceRenderer(reader, styler)),
        subjects=subjects,
        language=lang,
        dataset=dataset,
    )


def _evalplus_instruct(
    id: str, locale: MbppLocale, subjects: ListOfSubjects, lang: Language, dataset: DatasetPolicy
) -> Benchmark:
    return execution(
        id,
        dataset_path=MBPP_ELLAMIND_DATASET_PATH,
        instruction=_MbppInstruction(locale).evalplus,
        cue="",  # no forced preamble; the code block is extracted from the free-form response
        fewshot_target=lambda demo: f"```python\n{_norm_code(demo)}\n```",
        answer=ReconstructProgram(instruct_reconstruct, stop_sequences=[]),
        fewshot_source=SampleSplit(),
        subjects=subjects,
        language=lang,
        dataset=dataset,
    )


_BUILDERS: dict[TaskVariant, _Builder] = {
    "OLMES": _olmes,
    "BPB_OLMES": _bpb_olmes,
    "EvalPlus": _evalplus,
    "BPB_EvalPlus": _bpb_evalplus,
    "EvalPlusInstruct": _evalplus_instruct,
}


def mbpp_ellamind(task_variant: TaskVariant, language: LanguageCode, dataset: DatasetPolicy | None = None) -> Benchmark:
    """Build ``MBPP{language}_{variant}``, e.g. ``mbpp_ellamind("OLMES", "DE")``."""
    lang, config = ellamind_language(language)
    dataset_policy = dataset if dataset is not None else pinned_by_framework(MBPP_ELLAMIND_DATASET_PATH)
    return _BUILDERS[task_variant](
        _ID_TEMPLATES[task_variant].format(language=language),
        _LOCALES[language],
        ListOfSubjects([config]),
        lang,
        dataset_policy,
    )


MBPP_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    mbpp_ellamind(task_variant, language) for language in _LANGUAGES for task_variant in get_args(TaskVariant)
]
