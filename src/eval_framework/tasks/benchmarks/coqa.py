from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.f1 import F1
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.utils import get_n_letters


def _expand_coqa_doc(doc: dict[str, Any]) -> list[dict[str, Any]]:
    questions = doc.get("questions") or {}
    q_list = questions.get("input_text") or []
    answers = doc.get("answers") or {}
    a_list = answers.get("input_text") or []
    additional = doc.get("additional_answers") or {}
    out = []
    for turn_idx in range(len(q_list)):
        ans_at_turn = [a_list[turn_idx]] if turn_idx < len(a_list) else []
        for add in additional.values():
            if isinstance(add, dict):
                add_text = add.get("input_text") or []
                if turn_idx < len(add_text):
                    ans_at_turn.append(add_text[turn_idx])
        prev_qa = []
        for i in range(turn_idx):
            if i < len(q_list) and i < len(a_list):
                prev_qa.append((q_list[i], a_list[i]))
        out.append({
            "id": f"{doc.get('id', '')}_turn{turn_idx}",
            "story": doc.get("story", ""),
            "source": doc.get("source", ""),
            "question": q_list[turn_idx],
            "answers": ans_at_turn,
            "previous_qa": prev_qa,
        })
    return out


class CoQA(BaseTask[str]):
    """CoQA schema: id, source, story, questions, answers, additional_answers. One sample per turn."""

    NAME = "CoQA"
    DATASET_PATH = "EleutherAI/coqa"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion, F1]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Passage"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["\n\n"]
        self.max_tokens = 50

    def _load_dataset(self, subject: str) -> None:
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH)
        expanded = []
        for split in (self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT):
            if split not in hf_dataset:
                continue
            for doc in hf_dataset[split]:
                expanded.extend(_expand_coqa_doc(doc))
        by_split = {self.SAMPLE_SPLIT: expanded, self.FEWSHOT_SPLIT: expanded}
        self.dataset = self._shuffle_splits(hf_dataset=by_split)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        parts = [f"Passage: {item.get('story', '')}"]
        prev = item.get("previous_qa") or []
        if prev:
            parts.append("\n\nPreceding questions:")
            for q, a in prev:
                parts.append(f"\n\nQuestion: {q}\nAnswer: {a}")
        parts.append("\n\nFinal question:")
        parts.append(f"\n\nQuestion: {item.get('question', '')}\n")
        return "".join(parts)

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        return item.get("answers") or []

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        gt = self._get_ground_truth(item)
        return f" {gt[0]}" if gt else ""

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        return completion_text.strip()


class CoQAMC(CoQA):
    """CoQA multiple choice variant."""

    NAME = "CoQAMC"
    DATASET_PATH = "allenai/coqa-gen2mc"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG

    def _load_dataset(self, subject: str) -> None:
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH)
        expanded = []
        for split in (self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT):
            if split not in hf_dataset:
                continue
            for doc in hf_dataset[split]:
                expanded.append(doc)
        by_split = {self.SAMPLE_SPLIT: expanded, self.FEWSHOT_SPLIT: expanded}
        self.dataset = self._shuffle_splits(hf_dataset=by_split)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        parts = [f"Passage: {item.get('story', '')}"]
        prev = item.get("previous_qa") or []
        if prev:
            parts.append("\n\nPreceding questions:")
            for q, a in prev:
                parts.append(f"\n\nQuestion: {q}\nAnswer: {a}")
        parts.append("\n\nFinal question:")
        parts.append(f"\n\nQuestion: {item.get('question', '')}")
        choices = item.get("choices", [])
        if choices:
            parts.append("\nChoices:")
            for idx, c in enumerate(choices):
                parts.append(f"\n{chr(ord('A')+idx)}. {c}")
        parts.append("\n")
        return "".join(parts)

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        gt = item.get("answer")
        if isinstance(gt, str):
            return [gt]
        elif isinstance(gt, list):
            return gt
        return []

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        gt = self._get_ground_truth(item)
        return f" {gt[0]}" if gt else ""

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        stripped = completion_text.strip()
        if sample and 'choices' in sample.input and stripped in sample.input['choices']:
            idx = sample.input['choices'].index(stripped)
            return chr(ord('A') + idx)
        if len(stripped) == 1 and stripped.upper() in ['A', 'B', 'C', 'D', 'E']:
            return stripped.upper()
        return stripped
