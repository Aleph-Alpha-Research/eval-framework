"""MMLU-ProX: multilingual MMLU-Pro.

https://huggingface.co/datasets/li-lab/MMLU-ProX
"""

import re
from typing import Any, final, override

from datasets import DatasetDict

from eval_framework.answer import AnswerPolicy, ExtractFromCompletion, first_match, last_match
from eval_framework.benchmarks.cot import CotPrompt
from eval_framework.benchmarks.mmlu_pro import MMLU_PRO_SUBJECTS
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import EvalKind, SampleBody, assemble_messages
from eval_framework.fewshot import FewShot, FewshotExample, FewShotPolicy, FewShotSplit, FunctionRenderer, NoFewShot
from eval_framework.metrics.base import BaseMetric
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, InitialPrompt, MCStyle, TaskStyler
from eval_framework.tasks.utils import get_n_letters
from template_formatting.formatter import Message

_MMLU_MAX_NUM_OPTIONS = 10  # MMLU-Pro(X) has at most 10 options


@final
class MMLUProXReader(ChoiceReader):
    """Reads an MMLU-ProX item, returns question, up to 10 answer options, and the correct index.

    Options are stored flat as ``option_0`` .. ``option_9``. Rows with fewer than 10 options are
    padded, with ``None`` in the test split but the literal string ``"N/A"`` in the validation
    split; both are dropped here, matching the official MMLU-Pro evaluation code (the
    lm-evaluation-harness MMLU-ProX tasks drop only ``None``, so their fewshot examples show
    "N/A" as an answer option). We deviate from this by dropping both ``None`` and ``"N/A"``.
    """

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        raw_options = [item[f"option_{i}"] for i in range(_MMLU_MAX_NUM_OPTIONS)]
        # Drop both None and "N/A"
        choices = [option for option in raw_options if option is not None and option != "N/A"]
        # BUG-HINT: on one MMLU-ProX test row (question_id 3787, chemistry, identical in the `en` and `de`
        # configs) ``answer_index`` is 1 while the ``answer`` column says "C", and the question's
        # electronegativities support the answer column. The choice benchmarks built on this index score
        # that row against a wrong gold; the CoT variants read the ``answer`` column and are unaffected.
        correct_index = item["answer_index"]
        # Assert that padding is trailing, otherwise it would shift answer_index off the answer.
        assert raw_options[: len(choices)] == choices, f"Non-trailing padded option in item {item['question_id']}"
        assert 0 <= correct_index < len(choices), f"answer_index points at a padded option in {item['question_id']}"
        return ChoiceFields(raw_question=item["question"], choices=choices, correct_index=correct_index)


@final
class MMLUProXDatasetLoader(DatasetLoader):
    """Loads one MMLU-ProX language and filters its rows to one category.

    MMLU-ProX has two subject axes, but a benchmark's subjects can only carry one: the language
    is fixed per benchmark instance (it is the HuggingFace config the wrapped loader downloads),
    while the ``name`` this loader receives from the subject selection is the category to filter
    rows by. Both splits are filtered, so fewshot examples are category-matched.
    """

    def __init__(self, language: str, inner: DatasetLoader) -> None:
        self._language = language
        self._inner = inner

    @override
    def load(self, name: str | None) -> DatasetDict:
        assert name is not None, "MMLU-ProX subjects are the categories; a category must be selected"
        dataset = self._inner.load(self._language)
        return dataset.filter(lambda row: row["category"] == name)

    @override
    def metadata(self) -> dict[str, str]:
        return self._inner.metadata()


@final
class MMLUProXDatasetPolicy(DatasetPolicy):
    """Pins an MMLU-ProX dataset via the framework lockfile; its loaders load one language."""

    def __init__(self, dataset_path: str, language: str) -> None:
        self._pinned = pinned_by_framework(dataset_path)
        self._language = language

    @override
    def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
        return MMLUProXDatasetLoader(self._language, self._pinned.loader(custom_hf_revision))

    @override
    def documentation(self) -> str:
        return self._pinned.documentation()


# Supported languages: HF config -> Language enum. Extend together with the styler blocks below.
_MMLU_PROX_LANGUAGES = {"de": Language.DEU}

# German category names, following the lm-evaluation-harness MMLU-ProX/de task descriptions.
_CATEGORY_NAMES_DE = {
    "biology": "Biologie",
    "business": "Wirtschaft",
    "chemistry": "Chemie",
    "computer science": "Informatik",
    "economics": "Ökonomie",
    "engineering": "Ingenieurwesen",
    "health": "Gesundheit",
    "history": "Geschichte",
    "law": "Recht",
    "math": "Mathematik",
    "other": "Sonstiges",
    "philosophy": "Philosophie",
    "physics": "Physik",
    "psychology": "Psychologie",
}


def _mc_preamble_de(subject: str) -> str:
    return f"Die folgenden sind Multiple-Choice-Fragen (mit Antworten) über {_CATEGORY_NAMES_DE[subject]}."


def _cloze_preamble_de(subject: str) -> str:
    # Cloze/BPB prompts show no answer options, so the preamble must not claim multiple choice.
    return f"Die folgenden sind Fragen (mit Antworten) über {_CATEGORY_NAMES_DE[subject]}."


def _mmlu_prox_benchmark(
    id: str,
    styler: TaskStyler,
    language: str,
    lite: bool = False,
    dataset: DatasetPolicy | None = None,
) -> Benchmark:
    dataset_path = "li-lab/MMLU-ProX-Lite" if lite else "li-lab/MMLU-ProX"
    dataset_policy = dataset if dataset is not None else MMLUProXDatasetPolicy(dataset_path, language)
    return ComposedBenchmark.choice(
        id=id,
        reader=MMLUProXReader(),
        styler=styler,
        sample_split="test",
        fewshot_split="validation",
        # The subjects are the 14 MMLU-Pro categories (identical in MMLU-ProX)
        subjects=ListOfSubjects(MMLU_PRO_SUBJECTS),
        # The language axis lives in the dataset policy so that we can select the right question/answer cues.
        dataset_policy=dataset_policy,
        language=_MMLU_PROX_LANGUAGES[language],
    )


def mmlu_prox_lilab_de_mc(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _mmlu_prox_benchmark(
        "MMLUProX_LiLab_DE_MC",
        MCStyle.for_language(Language.DEU, initial_prompt=_mc_preamble_de),
        "de",
        dataset=dataset,
    )


def mmlu_prox_lilab_de_mc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _mmlu_prox_benchmark(
        "MMLUProX_LiLab_DE_MC_OLMES",
        MCStyle.for_language(Language.DEU, space_prefixed_labels=True, initial_prompt=_mc_preamble_de),
        "de",
        dataset=dataset,
    )


def mmlu_prox_lite_lilab_de_mc(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _mmlu_prox_benchmark(
        "MMLUProXLite_LiLab_DE_MC",
        MCStyle.for_language(Language.DEU, initial_prompt=_mc_preamble_de),
        "de",
        lite=True,
        dataset=dataset,
    )


def mmlu_prox_lite_lilab_de_mc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _mmlu_prox_benchmark(
        "MMLUProXLite_LiLab_DE_MC_OLMES",
        MCStyle.for_language(Language.DEU, space_prefixed_labels=True, initial_prompt=_mc_preamble_de),
        "de",
        lite=True,
        dataset=dataset,
    )


def mmlu_prox_lilab_de_cloze(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _mmlu_prox_benchmark(
        "MMLUProX_LiLab_DE_CLOZE",
        ClozeStyle.for_language(Language.DEU, initial_prompt=_cloze_preamble_de),
        "de",
        dataset=dataset,
    )


def mmlu_prox_lite_lilab_de_cloze(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _mmlu_prox_benchmark(
        "MMLUProXLite_LiLab_DE_CLOZE",
        ClozeStyle.for_language(Language.DEU, initial_prompt=_cloze_preamble_de),
        "de",
        lite=True,
        dataset=dataset,
    )


def mmlu_prox_lilab_de_bpb(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _mmlu_prox_benchmark(
        "MMLUProX_LiLab_DE_BPB",
        BPBStyle.for_language(Language.DEU, initial_prompt=_cloze_preamble_de),
        "de",
        dataset=dataset,
    )


def mmlu_prox_lite_lilab_de_bpb(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _mmlu_prox_benchmark(
        "MMLUProXLite_LiLab_DE_BPB",
        BPBStyle.for_language(Language.DEU, initial_prompt=_cloze_preamble_de),
        "de",
        lite=True,
        dataset=dataset,
    )


# --- chain-of-thought (generative) variants ------------------------------------------------------------------

# Cue closing every v1 prompt, and the prefix the dataset's ``cot_content`` starts with instead of it.
_COT_CUE_DE = "Antwort: Denken wir Schritt für Schritt nach."
_COT_CONTENT_PREFIX_DE = "A: Denken wir Schritt für Schritt nach."
_COT_ANSWER_RE_DE = re.compile(r"Die Antwort ist \(?([ABCDEFGHIJ])\)?")
_COT_ANSWER_RE_DE_V2 = re.compile(
    rf"\b(?:Antwort\s+ist|answer\s+is):?\s*\(?([{''.join(get_n_letters(_MMLU_MAX_NUM_OPTIONS))}])\b\)?",
    re.IGNORECASE,
)


def _cot_preamble_de(subject: str) -> str:
    return (
        f"{_mc_preamble_de(subject)} Denken Sie Schritt für Schritt nach und beenden Sie Ihre Antwort mit "
        '"Die Antwort ist (X)", wobei X der richtige Buchstabe ist.'
    )


def _cot_preamble_de_v2(subject: str) -> str:
    return (
        f"{_mc_preamble_de(subject)} Geben Sie eine KNAPPE Begründung und beenden Sie Ihre Antwort mit "
        '"Die Antwort ist (X)", wobei X der richtige Buchstabe ist.'
    )


def _cot_body_de(raw_question: str, choices: list[str]) -> str:
    options = "".join(f"{key}. {choice}\n" for key, choice in zip(get_n_letters(len(choices)), choices))
    return f"Frage:\n{raw_question}\nOptionen:\n{options}"


def _cot_body_de_v2(raw_question: str, choices: list[str]) -> str:
    # "(A) ..."-labelled options, matching the "(X)" answer format the v2 preamble asks for.
    options = "".join(f"({key}) {choice}\n" for key, choice in zip(get_n_letters(len(choices)), choices))
    return f"Frage:\n{raw_question}\nOptionen:\n{options}"


@final
class _AnswerColumnCot(EvalKind):
    """Multiple-choice chain-of-thought over MMLU-ProX rows, scored by accuracy on the extracted letter.

    Differs from the shared ``Cot`` kind in two ways: the gold letter is the dataset's ``answer`` column
    rather than derived from ``answer_index`` (the two disagree on one test row, and the answer column is
    the correct one — see ``MMLUProXReader``), and the assistant cue is configurable (``""`` for none)."""

    def __init__(self, reader: ChoiceReader, *, build_prompt: CotPrompt, preamble: InitialPrompt, cue: str) -> None:
        self._reader = reader
        self._build_prompt = build_prompt
        self._preamble = preamble
        self._cue = cue

    @override
    def metrics(self) -> list[type[BaseMetric]]:
        return [AccuracyCompletion]

    @override
    def samples(self, item: dict[str, Any]) -> list[SampleBody]:
        fields = self._reader.read(item)
        return [
            SampleBody(
                prompt=self._build_prompt(fields.raw_question, fields.choices),
                cue=self._cue,
                possible_completions=[],
                ground_truth=item["answer"],  # the bare answer letter
            )
        ]

    @override
    def messages(self, body: SampleBody, *, fewshot: list[FewshotExample], subject_label: str) -> list[Message]:
        return assemble_messages(fewshot, body, initial_prompt=self._preamble(subject_label))


def _cot_fewshot_de() -> FewShot:
    # Demonstrations answer with the row's worked reasoning, its "A:" prefix replaced by the cue (keeping
    # the dataset's missing whitespace after it).
    def demo(item: dict[str, Any]) -> FewshotExample:
        fields = MMLUProXReader().read(item)
        return FewshotExample(
            prompt=_cot_body_de(fields.raw_question, fields.choices),
            answer=item["cot_content"].replace(_COT_CONTENT_PREFIX_DE, _COT_CUE_DE),
        )

    return FewShot(FewShotSplit("validation"), FunctionRenderer(demo))


def _mmlu_prox_cot_benchmark(
    id: str,
    kind: EvalKind,
    answer: AnswerPolicy,
    fewshot: FewShotPolicy,
    lite: bool,
    dataset: DatasetPolicy | None,
) -> Benchmark:
    dataset_path = "li-lab/MMLU-ProX-Lite" if lite else "li-lab/MMLU-ProX"
    dataset_policy = dataset if dataset is not None else MMLUProXDatasetPolicy(dataset_path, "de")
    return ComposedBenchmark.compose(
        id=id,
        kind=kind,
        answer=answer,
        sample_split="test",
        fewshot=fewshot,
        subjects=ListOfSubjects(MMLU_PRO_SUBJECTS),
        dataset_policy=dataset_policy,
        language=Language.DEU,
    )


def mmlu_prox_lilab_de_cot(dataset: DatasetPolicy | None = None) -> Benchmark:
    """MMLU-ProX German, 5-shot chain-of-thought (generative), mirroring lm-evaluation-harness
    ``lm_eval/tasks/mmlu_prox/de`` for result parity. Run with ``num_fewshot=5`` for parity.

    Differences to the lm-eval implementation (all deliberate):

    - Preamble: lm-eval says "Im Folgenden sind Multiple-Choice-Fragen (mit Antworten) zu {X}.";
      we keep the GlobalMMLU-style first sentence of the loglikelihood siblings ("Die folgenden
      sind ... über {X}.") and append lm-eval's instruction sentence verbatim.
    - Preamble separator: the framework puts a blank line between preamble and first question,
      lm-eval a single newline.
    - "N/A" option padding: ``MMLUProXReader`` drops it, lm-eval drops only ``None``,
      so 21 of its 70 fewshot examples list "N/A" as an answer option.
    - Fewshot selection: lm-eval takes the first n validation rows, we sample randomly (seeded).
      The validation split holds exactly 5 rows per category, so with 5-shot both use the same
      rows and only the order differs.
    - The cue is a separate assistant message (a prefill under chat templates); lm-eval appends it
      to the user text. Under ``ConcatFormatter`` both render identically.
    - Metric: lm-eval scores ``exact_match`` with ignore_case/ignore_punctuation on the extracted
      letter; the regex only ever yields one uppercase letter, so plain ``AccuracyCompletion``
      is equivalent.
    - Test-split order is shuffled by the framework (irrelevant for full runs).
    - Token budget: lm-eval caps generation at 2048 tokens; ``max_tokens`` stays unset, because
      reasoning models routinely think past 2048 tokens and the cap turned 35-52% of their
      completions into length terminations.
    - Stop sequences: lm-eval stops on "</s>", "Q:", "Frage:" and "<|im_end|>"; none are set here.
      This benchmark is meant for post-trained models, which end their turn via EOS anyway, and a
      "Frage:" stop would truncate a reasoning model that restates the question mid-thought.

    Identical: per-example layout ("Frage:\\n...\\nOptionen:\\nA. ...\\n"), fewshot targets from the
    row's ``cot_content`` with its "A:" prefix replaced by the cue (including the dataset's missing
    whitespace after the cue), and extraction regex.

    Differences to eval-framework's ``MMLU_PRO_COT`` (TIGER-Lab/MMLU-Pro, English): that task uses
    the Tülu-3 zero-shot prompt with "(A)"-style options, the regex "Therefore, the answer is (X)"
    and the stop sequence "Question:", and forbids fewshots.
    """
    kind = _AnswerColumnCot(MMLUProXReader(), build_prompt=_cot_body_de, preamble=_cot_preamble_de, cue=_COT_CUE_DE)
    answer = ExtractFromCompletion(first_match(_COT_ANSWER_RE_DE))
    return _mmlu_prox_cot_benchmark("MMLUProX_LiLab_DE_COT", kind, answer, _cot_fewshot_de(), False, dataset)


def mmlu_prox_lite_lilab_de_cot(dataset: DatasetPolicy | None = None) -> Benchmark:
    """``mmlu_prox_lilab_de_cot`` on the MMLU-ProX-Lite subset (588 German test rows, same validation split)."""
    kind = _AnswerColumnCot(MMLUProXReader(), build_prompt=_cot_body_de, preamble=_cot_preamble_de, cue=_COT_CUE_DE)
    answer = ExtractFromCompletion(first_match(_COT_ANSWER_RE_DE))
    return _mmlu_prox_cot_benchmark("MMLUProXLite_LiLab_DE_COT", kind, answer, _cot_fewshot_de(), True, dataset)


def mmlu_prox_lilab_de_cot_v2(dataset: DatasetPolicy | None = None) -> Benchmark:
    """``mmlu_prox_lilab_de_cot`` aligned with ``MMLU_PRO_COT_V2``: no assistant cue, lenient last-match
    extraction, 0-shot only.

    The v1 cue is an assistant prefill. Chat servers close it as a finished turn, so reasoning models answer
    in a fresh turn and some skip closing their think block, which leaves the answer outside the scored
    content. Some chat templates reject the trailing assistant message outright.

    Also aligned with ``MMLU_PRO_COT_V2``'s prompt: the preamble asks for CONCISE reasoning rather than
    step-by-step thinking, and options are labelled "(A) ..." to match the "(X)" answer format. v1 keeps
    lm-eval's wording and "A. ..." labels for parity.
    """
    kind = _AnswerColumnCot(MMLUProXReader(), build_prompt=_cot_body_de_v2, preamble=_cot_preamble_de_v2, cue="")
    answer = ExtractFromCompletion(last_match(_COT_ANSWER_RE_DE_V2))
    return _mmlu_prox_cot_benchmark("MMLUProX_LiLab_DE_COT_V2", kind, answer, NoFewShot(), False, dataset)


def mmlu_prox_lite_lilab_de_cot_v2(dataset: DatasetPolicy | None = None) -> Benchmark:
    """``mmlu_prox_lilab_de_cot_v2`` on the MMLU-ProX-Lite subset (588 German test rows, same validation split)."""
    kind = _AnswerColumnCot(MMLUProXReader(), build_prompt=_cot_body_de_v2, preamble=_cot_preamble_de_v2, cue="")
    answer = ExtractFromCompletion(last_match(_COT_ANSWER_RE_DE_V2))
    return _mmlu_prox_cot_benchmark("MMLUProXLite_LiLab_DE_COT_V2", kind, answer, NoFewShot(), True, dataset)


MMLU_PROX_BENCHMARKS: list[Benchmark] = [
    mmlu_prox_lilab_de_mc(),
    mmlu_prox_lilab_de_mc_olmes(),
    mmlu_prox_lilab_de_cloze(),
    mmlu_prox_lilab_de_bpb(),
    mmlu_prox_lilab_de_cot(),
    mmlu_prox_lilab_de_cot_v2(),
    mmlu_prox_lite_lilab_de_mc(),
    mmlu_prox_lite_lilab_de_mc_olmes(),
    mmlu_prox_lite_lilab_de_cloze(),
    mmlu_prox_lite_lilab_de_bpb(),
    mmlu_prox_lite_lilab_de_cot(),
    mmlu_prox_lite_lilab_de_cot_v2(),
]
