from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


def _flatten_validated_answers(validated_answers: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten validated_answers from dict of lists to list of dicts."""
    num_list = validated_answers.get("number") or []
    date_list = validated_answers.get("date") or []
    spans_list = validated_answers.get("spans") or []
    n = max(len(num_list), len(date_list), len(spans_list))
    out = []
    for i in range(n):
        out.append(
            {
                "number": num_list[i] if i < len(num_list) else "",
                "date": date_list[i] if i < len(date_list) else {"day": "", "month": "", "year": ""},
                "spans": spans_list[i] if i < len(spans_list) else [],
            }
        )
    return out


def _parse_answer(answer: dict[str, Any]) -> tuple[str, ...]:
    """Return a hashable tuple for one answer (number, spans, or date string)."""
    if answer.get("number"):
        return (str(answer["number"]),)
    spans = answer.get("spans") or []
    if spans:
        return tuple(spans)
    date = answer.get("date") or {}
    day = date.get("day") or ""
    month = date.get("month") or ""
    year = date.get("year") or ""
    return (" ".join([day, month, year]).strip(),)


def _get_answers(doc: dict[str, Any]) -> list[tuple[str, ...]]:
    """Deduplicated list of valid answer tuples (main answer + validated_answers)."""
    answer = doc.get("answer") or {}
    validated = doc.get("validated_answers") or {}
    candidates = [answer] + _flatten_validated_answers(validated)
    seen: set[tuple[str, ...]] = set()
    out = []
    for cand in candidates:
        if not cand:
            continue
        parsed = _parse_answer(cand)
        if parsed in seen or (len(parsed) == 1 and parsed[0] == ""):
            continue
        seen.add(parsed)
        out.append(parsed)
    return out


def _tuple_to_display(tup: tuple[str, ...]) -> str:
    """Single string for loglikelihood prompt (space-prefixed for cue)."""
    if not tup:
        return ""
    if len(tup) == 1:
        return str(tup[0])
    return ", ".join(str(x) for x in tup)


class DropRC(BaseTask[str]):
    """Original DROP (EleutherAI/drop): passage, question, answer + validated_answers.

    Parsed answers used as choices for rank-choice.
    """

    NAME = "DropRC"
    DATASET_PATH = "EleutherAI/drop"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Passage"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(5)

    def _load_dataset(self, subject: str) -> None:
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH)
        validation = list(hf_dataset.get(self.SAMPLE_SPLIT, []))
        processed = []
        for doc in validation:
            parsed = _get_answers(doc)
            if not parsed:
                continue
            processed.append({**doc, "parsed_answers": parsed})
        self.dataset = self._shuffle_splits(hf_dataset={self.SAMPLE_SPLIT: processed, self.FEWSHOT_SPLIT: processed})

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        passage = (item.get("passage") or "").strip()
        question = item.get("question", "")
        return f"Passage: {passage}\nQuestion: {question}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        answers = item.get("parsed_answers") or []
        if not answers:
            return None
        return f" {_tuple_to_display(answers[0])}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        answers = item.get("parsed_answers") or []
        if not answers:
            return None
        return [f" {_tuple_to_display(t)}" for t in answers]


class DropMC(BaseTask[str]):
    """Multiple-choice variant using allenai/drop-gen2mc (passage_original, question_original, choices, answerKey)."""

    NAME = "DropMC"
    DATASET_PATH = "allenai/drop-gen2mc"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Passage"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(5)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        passage = (item.get("passage_original") or "").strip()
        question = item.get("question_original", "")
        texts = item.get("choices", {}).get("text", [])
        labels = item.get("choices", {}).get("label", self.keys[: len(texts)])
        options = "\n".join(f"{label}. {t}" for label, t in zip(labels, texts))
        return f"Passage: {passage}\nQuestion: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        key = item.get("answerKey", "A")
        return f" {key}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        labels = item.get("choices", {}).get("label", [])
        return [f" {label}" for label in labels]
