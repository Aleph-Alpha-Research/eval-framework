"""MMLU-ProX: multilingual MMLU-Pro.

https://huggingface.co/datasets/li-lab/MMLU-ProX

Each language is one config of the dataset ("de", "fr", "it", "es", "pt", "ru", "uk", "hu"); its
subjects are the 14 MMLU-Pro categories. The Lite variants run on ``li-lab/MMLU-ProX-Lite`` (a 588-row
test subset per language, same validation split).

Variants, per language and for both the full and the Lite dataset:

- ``MC`` / ``MC_OLMES``: multiple choice, scored over the letter labels (OLMES: space-prefixed labels).
- ``Cloze`` / ``BPB``: options hidden, scored over the option texts, or BPB of the gold option.

Adding a language: extend ``LanguageCode`` and add its wording to ``_LOCALES``. The chain-of-thought variants
(``COT`` / ``COT_V2``) are German only and built separately.
"""

import re
from dataclasses import dataclass
from typing import Any, Literal, assert_never, final, get_args, override

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
from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, InitialPrompt, Language, MCStyle, TaskStyler
from eval_framework.tasks.utils import get_n_letters
from template_formatting.formatter import Message

_MMLU_MAX_NUM_OPTIONS = 10  # MMLU-Pro(X) has at most 10 options

TaskVariant = Literal["MC", "MC_OLMES", "Cloze", "BPB"]

LanguageCode = Literal["DE", "FR", "IT", "ES", "PT", "RU", "UK", "HU"]
_LANGUAGES: tuple[LanguageCode, ...] = get_args(LanguageCode)


def _task_id(task_variant: TaskVariant, language: LanguageCode, lite: bool) -> str:
    return f"MMLUProX{'Lite' if lite else ''}_LiLab_{language}_{task_variant}"


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


@dataclass(frozen=True, kw_only=True)
class MMLUProXLocale:
    """One language of MMLU-ProX: its dataset config, and the wording of the choice-variant preambles."""

    config: str
    """The HuggingFace config holding the language (``de``)."""
    language: Language
    """The framework language; also selects the question/answer cues of the choice stylers."""
    category_names: dict[str, str]
    """MMLU-Pro category -> its name in the language, as filled into ``{category}`` of the preambles."""
    mc_template: str
    """Opens a prompt that shows the answer options; ``{category}`` is the category name."""
    cloze_template: str
    """Opens a prompt that hides the answer options, so it must not claim multiple choice."""

    def __post_init__(self) -> None:
        if set(self.category_names) != set(MMLU_PRO_SUBJECTS):
            raise ValueError(f"category_names must name exactly the MMLU-Pro categories, got {self.category_names}")
        for preamble in (self.mc_template, self.cloze_template):
            if "{category}" not in preamble:
                raise ValueError(f"preamble must contain '{{category}}': {preamble!r}")

    def mc_preamble(self, subject: str) -> str:
        return self.mc_template.format(category=self.category_names[subject])

    def cloze_preamble(self, subject: str) -> str:
        return self.cloze_template.format(category=self.category_names[subject])


# Category names are taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/mmlu_prox/lang_libs.py
_LOCALES: dict[LanguageCode, MMLUProXLocale] = {
    "DE": MMLUProXLocale(
        config="de",
        language=Language.DEU,
        # Following the lm-evaluation-harness MMLU-ProX/de task descriptions.
        category_names={
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
        },
        mc_template="Die folgenden sind Multiple-Choice-Fragen (mit Antworten) über {category}.",
        cloze_template="Die folgenden sind Fragen (mit Antworten) über {category}.",
    ),
    "FR": MMLUProXLocale(
        config="fr",
        language=Language.FRA,
        category_names={
            "biology": "biologie",
            "business": "commerce",
            "chemistry": "chimie",
            "computer science": "informatique",
            "economics": "économie",
            "engineering": "ingénierie",
            "health": "santé",
            "history": "histoire",
            "law": "droit",
            "math": "mathématiques",
            "other": "autre",
            "philosophy": "philosophie",
            "physics": "physique",
            "psychology": "psychologie",
        },
        mc_template="Voici des questions à choix multiples (avec réponses) concernant {category}.",
        cloze_template="Voici des questions (avec réponses) concernant {category}.",
    ),
    "IT": MMLUProXLocale(
        config="it",
        language=Language.ITA,
        category_names={
            "biology": "biologia",
            "business": "affari",
            "chemistry": "chimica",
            "computer science": "informatica",
            "economics": "economia",
            "engineering": "ingegneria",
            "health": "salute",
            "history": "storia",
            "law": "diritto",
            "math": "matematica",
            "other": "altro",
            "philosophy": "filosofia",
            "physics": "fisica",
            "psychology": "psicologia",
        },
        mc_template="Le seguenti sono domande a scelta multipla (con risposte) riguardanti {category}.",
        cloze_template="Di seguito sono riportate delle domande (con risposte) relative a {category}.",
    ),
    "ES": MMLUProXLocale(
        config="es",
        language=Language.SPA,
        # Question/answer cues: "Pregunta:" / "Respuesta:"
        category_names={
            "biology": "biología",
            "business": "negocios",
            "chemistry": "química",
            "computer science": "informática",
            "economics": "economía",
            "engineering": "ingeniería",
            "health": "salud",
            "history": "historia",
            "law": "derecho",
            "math": "matemáticas",
            "other": "otro",
            "philosophy": "filosofía",
            "physics": "física",
            "psychology": "psicología",
        },
        mc_template="Las siguientes son preguntas de opción múltiple (con respuestas) sobre {category}.",
        cloze_template="Las siguientes son preguntas (con respuestas) sobre {category}.",
    ),
    "PT": MMLUProXLocale(
        config="pt",
        language=Language.POR,
        # Question/answer cues: "Pergunta:" / "Resposta:"
        category_names={
            "biology": "biologia",
            "business": "negócios",
            "chemistry": "química",
            "computer science": "ciência da computação",
            "economics": "economia",
            "engineering": "engenharia",
            "health": "saúde",
            "history": "história",
            "law": "direito",
            "math": "matemática",
            "other": "outro",
            "philosophy": "filosofia",
            "physics": "física",
            "psychology": "psicologia",
        },
        mc_template="As seguintes são questões de múltipla escolha (com respostas) sobre {category}.",
        cloze_template="As seguintes são perguntas (com respostas) sobre {category}.",
    ),
    "RU": MMLUProXLocale(
        config="ru",
        language=Language.RUS,
        # Question/answer cues: "Вопрос:" / "Ответ:"
        category_names={
            "biology": "Биология",
            "business": "Бизнес",
            "chemistry": "Химия",
            "computer science": "Информатика",
            "economics": "Экономика",
            "engineering": "Инженерия",
            "health": "Здравоохранение",
            "history": "История",
            "law": "Право",
            "math": "Математика",
            "other": "Другое",
            "philosophy": "Философия",
            "physics": "Физика",
            "psychology": "Психология",
        },
        mc_template="Ниже приведены вопросы с множественным выбором (с ответами) по теме {category}.",
        cloze_template="Ниже приведены вопросы (с ответами) по теме {category}.",
    ),
    "UK": MMLUProXLocale(
        config="uk",
        language=Language.UKR,
        # Question/answer cues: "Запитання:" / "Відповідь:"
        category_names={
            "biology": "біологія",
            "business": "бізнес",
            "chemistry": "хімія",
            "computer science": "інформатика",
            "economics": "економіка",
            "engineering": "інженерія",
            "health": "здоров'я",
            "history": "історія",
            "law": "право",
            "math": "математика",
            "other": "інше",
            "philosophy": "філософія",
            "physics": "фізика",
            "psychology": "психологія",
        },
        mc_template="Нижче наведено запитання з множинним вибором (з відповідями) щодо {category}.",
        cloze_template="Нижче наведено запитання (з відповідями) щодо {category}.",
    ),
    "HU": MMLUProXLocale(
        config="hu",
        language=Language.HUN,
        # Question/answer cues: "Kérdés:" / "Válasz:"
        category_names={
            "biology": "biológia",
            "business": "üzlet",
            "chemistry": "kémia",
            "computer science": "informatika",
            "economics": "közgazdaságtan",
            "engineering": "mérnöki tudományok",
            "health": "egészség",
            "history": "történelem",
            "law": "jog",
            "math": "matematika",
            "other": "egyéb",
            "philosophy": "filozófia",
            "physics": "fizika",
            "psychology": "pszichológia",
        },
        mc_template="Az alábbiak többválasztásos kérdések (válaszokkal) a {category} témakörben.",
        cloze_template="Az alábbiak a {category} témakörhöz kapcsolódó kérdések (válaszokkal együtt).",
    ),
}


# --- choice (loglikelihood) variants -------------------------------------------------------------------------


def _choice_benchmark(id: str, styler: TaskStyler, locale: MMLUProXLocale, dataset: DatasetPolicy) -> Benchmark:
    return ComposedBenchmark.choice(
        id=id,
        reader=MMLUProXReader(),
        styler=styler,
        sample_split="test",
        fewshot_split="validation",
        # The subjects are the 14 MMLU-Pro categories (identical in MMLU-ProX)
        subjects=ListOfSubjects(MMLU_PRO_SUBJECTS),
        # The language axis lives in the dataset policy so that we can select the right question/answer cues.
        dataset_policy=dataset,
    )


def _styler(task_variant: TaskVariant, locale: MMLUProXLocale) -> TaskStyler:
    match task_variant:
        case "MC":
            return MCStyle.for_language(locale.language, initial_prompt=locale.mc_preamble)
        case "MC_OLMES":
            return MCStyle.for_language(locale.language, space_prefixed_labels=True, initial_prompt=locale.mc_preamble)
        case "Cloze":
            return ClozeStyle.for_language(locale.language, initial_prompt=locale.cloze_preamble)
        case "BPB":
            return BPBStyle.for_language(locale.language, initial_prompt=locale.cloze_preamble)
        case _:
            assert_never(task_variant)


def mmlu_prox(
    task_variant: TaskVariant, language: LanguageCode, lite: bool = False, dataset: DatasetPolicy | None = None
) -> Benchmark:
    """Build ``MMLUProX[Lite]_LiLab_{language}_{variant}``, e.g. ``mmlu_prox("MC", "DE")``."""
    locale = _LOCALES[language]
    dataset_path = "li-lab/MMLU-ProX-Lite" if lite else "li-lab/MMLU-ProX"
    dataset_policy = dataset if dataset is not None else MMLUProXDatasetPolicy(dataset_path, locale.config)
    return _choice_benchmark(
        _task_id(task_variant, language, lite), _styler(task_variant, locale), locale, dataset_policy
    )


# --- chain-of-thought (generative) variants ------------------------------------------------------------------

_mc_preamble_de = _LOCALES["DE"].mc_preamble

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
    *(
        mmlu_prox(task_variant, language, lite=lite)
        for lite in (False, True)
        for language in _LANGUAGES
        for task_variant in get_args(TaskVariant)
    ),
    mmlu_prox_lilab_de_cot(),
    mmlu_prox_lilab_de_cot_v2(),
    mmlu_prox_lite_lilab_de_cot(),
    mmlu_prox_lite_lilab_de_cot_v2(),
]
