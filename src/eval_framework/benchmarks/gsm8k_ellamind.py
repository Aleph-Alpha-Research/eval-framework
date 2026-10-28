import re
from typing import Any, Literal, final, get_args, override

from eval_framework.answer import ExtractFromCompletion
from eval_framework.benchmarks.utils.ellamind_languages import ellamind_language
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Generative
from eval_framework.fewshot import ChoiceRenderer, FewShot, SampleSplit
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, question_cue_text

GSM8K_ELLAMIND_DATASET_PATH = "ellamind/gsm8k-platinum-multilingual"
_MAX_TOKENS = 1600

# The languages this dataset offers (see ``ellamind_languages`` for the token -> language/config table). The
# question prefix and answer cue are the language's defaults from ``task_style``; the final-answer line that
# closes a shown solution is the one GSM8K-specific string, so it is the only thing configured here.
LanguageCode = Literal["DE", "FR", "ES", "IT", "CS", "DA", "FI", "NL", "PL"]
_LANGUAGES: tuple[LanguageCode, ...] = get_args(LanguageCode)
_FINAL_ANSWER_LINE: dict[LanguageCode, str] = {
    "DE": "Daher ist die Antwort {answer}.",
    "FR": "La réponse est donc {answer}.",
    "ES": "Por lo tanto, la respuesta es {answer}.",
    "IT": "Pertanto, la risposta è {answer}.",
    "CS": "Odpověď je tedy {answer}.",
    "DA": "Derfor er svaret {answer}.",
    "FI": "Joten vastaus on {answer}.",
    "NL": "Dus het antwoord is {answer}.",
    "PL": "Zatem odpowiedź to {answer}.",
}


def _normalize_number(answer: str) -> str:
    """Drop thousands separators ('.' or ',') so a final answer is a bare integer string."""
    return answer.replace(".", "").replace(",", "")


@final
class _SolutionReader(ChoiceReader):
    """The single 'choice' is the gold solution followed by the language's final-answer line. A demonstration
    of the generative task shows the normalised integer; the BPB target keeps the raw final answer."""

    def __init__(self, final_answer_line: str, *, normalize_answer: bool) -> None:
        self._final_answer_line = final_answer_line
        self._normalize_answer = normalize_answer

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        final_answer = _normalize_number(item["final_answer"]) if self._normalize_answer else item["final_answer"]
        answer = f"{item['solution']} {self._final_answer_line.format(answer=final_answer)}"
        return ChoiceFields(raw_question=item["question"], choices=[answer], correct_index=0)


def _extract_final_integer(completion_text: str) -> str:
    """The last integer in the generation (EllaMind answers are integers), or ``"[invalid]"``."""
    numbers = re.findall(r"[-+]?\d+", _normalize_number(completion_text))
    return numbers[-1] if numbers else "[invalid]"


def _ellamind_dataset(dataset: DatasetPolicy | None) -> DatasetPolicy:
    return dataset if dataset is not None else pinned_by_framework(GSM8K_ELLAMIND_DATASET_PATH)


def gsm8k_ellamind_platinum(language: LanguageCode, dataset: DatasetPolicy | None = None) -> Benchmark:
    """Build ``GSM8K_Ellamind_{language}_Platinum``, e.g. ``gsm8k_ellamind_platinum("DE")``."""
    lang, config = ellamind_language(language)
    prefix, cue = question_cue_text(lang)
    # Demonstrations are sampled from the test split and rendered in the same answer format as a full
    # solution; the eval itself is free-form (the generative kind), so few-shot rendering and eval are
    # separate policies. Generation stops at the next question prefix (the start of a spurious demonstration).
    reader = _SolutionReader(_FINAL_ANSWER_LINE[language], normalize_answer=True)
    fewshot = FewShot(SampleSplit(), ChoiceRenderer(reader, ClozeStyle(question_prefix=prefix, cue_text=cue)))
    kind = Generative(
        build_prompt=lambda item: f"{prefix}{item['question']}\n",
        cue=cue,
        ground_truth=lambda item: _normalize_number(item["final_answer"]),
        metrics=[AccuracyCompletion],
    )
    return ComposedBenchmark.compose(
        id=f"GSM8K_Ellamind_{language}_Platinum",
        kind=kind,
        answer=ExtractFromCompletion(_extract_final_integer, [prefix.rstrip()], max_tokens=_MAX_TOKENS),
        sample_split="test",
        fewshot=fewshot,
        subjects=ListOfSubjects([config]),
        dataset_policy=_ellamind_dataset(dataset),
        language=lang,
    )


def gsm8k_ellamind_bpb_platinum(language: LanguageCode, dataset: DatasetPolicy | None = None) -> Benchmark:
    """Build ``GSM8K_Ellamind_{language}_BPB_Platinum``, e.g. ``gsm8k_ellamind_bpb_platinum("DE")``."""
    lang, config = ellamind_language(language)
    return ComposedBenchmark.choice(
        id=f"GSM8K_Ellamind_{language}_BPB_Platinum",
        reader=_SolutionReader(_FINAL_ANSWER_LINE[language], normalize_answer=False),
        styler=BPBStyle.for_language(lang),
        sample_split="test",
        fewshot_split="test",
        subjects=ListOfSubjects([config]),
        dataset_policy=_ellamind_dataset(dataset),
        language=lang,
    )


GSM8K_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    benchmark
    for language in _LANGUAGES
    for benchmark in (gsm8k_ellamind_platinum(language), gsm8k_ellamind_bpb_platinum(language))
]
