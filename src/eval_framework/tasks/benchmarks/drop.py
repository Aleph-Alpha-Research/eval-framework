from typing import Any

from eval_framework.metrics.completion.drop_completion import DropF1ExactMatch, DropMetricContext
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


def _flatten_validated_answers(validated_answers: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten validated_answers from dict of lists to list of dicts."""
    num_list = validated_answers.get("number") or []
    date_list = validated_answers.get("date") or []
    spans_list = validated_answers.get("spans") or []
    n = max(len(num_list), len(date_list), len(spans_list))
    return [
        {
            "number": num_list[i] if i < len(num_list) else "",
            "date": date_list[i] if i < len(date_list) else {"day": "", "month": "", "year": ""},
            "spans": spans_list[i] if i < len(spans_list) else [],
        }
        for i in range(n)
    ]


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
    return ", ".join(str(x) for x in tup) if tup else ""


class DropCompletion(BaseTask[str]):
    """DROP completion benchmark (EleutherAI/drop): passage, question, model generates answer.

    Uses DROP F1 and exact match. Stop at new paragraph or repeated prefixes.
    """

    NAME = "DropCompletion"
    DATASET_PATH = "EleutherAI/drop"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [DropF1ExactMatch]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Passage"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["Passage:", "Question:", "\n\n"]
        self.max_tokens = 50

    def _load_dataset(self, subject: str) -> None:
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH)

        def process(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            result = []
            for doc in docs:
                parsed = _get_answers(doc)
                if not parsed:
                    continue
                result.append({**doc, "parsed_answers": parsed})
            return result

        sample_split = process(hf_dataset.get(self.SAMPLE_SPLIT, []))
        fewshot_split = process(hf_dataset.get(self.FEWSHOT_SPLIT, []))
        self.dataset = self._shuffle_splits(
            hf_dataset={self.SAMPLE_SPLIT: sample_split, self.FEWSHOT_SPLIT: fewshot_split}
        )

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

    def _get_context(self, item: dict[str, Any]) -> DropMetricContext | None:
        answers = item.get("parsed_answers") or []
        if not answers:
            return None
        return DropMetricContext(answer_tuples=[list(a) for a in answers])

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"


class DropCompletion_OLMES(DropCompletion):
    """DropCompletion matching OLMES, using train split for fewshot and max tokens 100."""

    NAME = "DropCompletion_OLMES"
    FEWSHOT_SPLIT = "train"

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.max_tokens = 100

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        # TODO: Do we want this same prompt in the non_OLMES variant?
        context = (
            "The following are reading comprehension questions, where the answer to each question is either a "
            "segment of text from the corresponding passage, a number, or a date (containing any of the date, "
            "month, and/or year components). Some questions may require you to pull together information pieces "
            "from the passage and reason over them."
        )
        return context


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
        key = item.get("answerKey")
        if key is None:
            return None
        return f" {key}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        labels = item.get("choices", {}).get("label", [])
        return [f" {label}" for label in labels]

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"


class DropMC_OLMES(DropMC):
    """
    DropMC with OLMES-style prompt: space before each label in the prompt (" A.", " B.", ...).
    """

    NAME = "DropMC_OLMES"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        passage = (item.get("passage_original") or "").strip()
        question = item.get("question_original", "")
        texts = item.get("choices", {}).get("text", [])
        labels = item.get("choices", {}).get("label", self.keys[: len(texts)])
        options = "\n".join(f" {label}. {t}" for label, t in zip(labels, texts))
        return f"Passage: {passage}\nQuestion: {question}\n{options}\n"


class DropCloze(BaseTask[str]):
    """Cloze variant: loglikelihood ranking over full choice texts (allenai/drop-gen2mc).

    Same dataset as DropMC; options not shown in prompt; model scores full text of each choice.
    Includes BitsPerByte on the correct choice.
    """

    NAME = "DropCloze"
    DATASET_PATH = "allenai/drop-gen2mc"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Passage"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        passage = (item.get("passage_original") or "").strip()
        question = item.get("question_original", "")
        return f"Passage: {passage}\nQuestion: {question}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        texts = item.get("choices", {}).get("text", [])
        labels = item.get("choices", {}).get("label", [])
        key = item.get("answerKey")
        if key is None or key not in labels:
            return None
        idx = labels.index(key)
        return f" {texts[idx]}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        texts = item.get("choices", {}).get("text", [])
        return [f" {t}" for t in texts]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"
