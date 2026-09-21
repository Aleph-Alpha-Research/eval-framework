"""GSM8K (English) BaseTask base: https://huggingface.co/datasets/openai/gsm8k

Retained as the base class for the German EllaMind variant (``gsm8k_ellamind``), which is still a BaseTask.
The registered English variants (``GSM8K_OLMES``, ``GSM8KBPB``) are composed in ``benchmarks/gsm8k.py``.
"""

import re
from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


class GSM8KEvalHarness(BaseTask[str]):
    """GSM8K dataset: https://huggingface.co/datasets/openai/gsm8k
    This version uses samples from the train split as fewshot examples.
    """

    NAME = "GSM8KEvalHarness"
    DATASET_PATH = "openai/gsm8k"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    SUBJECTS = ["main"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)

        # until: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml
        self.stop_sequences: list[str] = ["Question:"]
        self.max_tokens = 1600

    def _extract_answer(self, completion: str) -> str:
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return "[invalid]"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]
        return self._extract_answer(completion_text)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['question']}\nAnswer:"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        return f" {item['answer']}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return self._extract_answer(item["answer"])
