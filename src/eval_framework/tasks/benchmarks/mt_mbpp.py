"""
Multilingual MBPP (MT MBPP). Code generation in multiple programming languages.

Dataset: allenai/multilingual_mbpp with one config per language (dataset name = language).
"""

from typing import Any
from eval_framework.tasks.base import Language
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.f1 import F1
from eval_framework.tasks.base import BaseTask, ResponseType, Sample

MT_MBPP_LANGUAGES = [
    "bash",
    "c",
    "cpp",
    "csharp",
    "go",
    "haskell",
    "java",
    "javascript",
    "matlab",
    "php",
    "python",
    "r",
    "ruby",
    "rust",
    "scala",
    "swift",
    "typescript",
]

BEGIN = "```"
END = "```"


class MTMBPP(BaseTask[str]):
    """
    Multilingual MBPP: generate code in a given language from a natural language prompt.
    One subject per programming language (dataset config name).
    """

    NAME = "MTMBPP"
    DATASET_PATH = "allenai/multilingual_mbpp"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "prompt"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    SUBJECTS = MT_MBPP_LANGUAGES
    PERTURBATION_UNMODIFIABLE_WORDS = None
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["\n\n"]
        self.max_tokens = 512

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return item["text"].strip()

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return f"\n{BEGIN}{item['language']}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | list[str] | None:
        code = item.get("code")
        if code is None:
            return None
        if isinstance(code, list):
            return [c.strip() if isinstance(c, str) else str(c) for c in code]
        return code.strip() if isinstance(code, str) else str(code)

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = item["code"]
        if isinstance(target, list):
            target = target[0] if target else ""
        lang = item.get("language", "python")
        return f"\n{BEGIN}{lang}\n{target.strip()}\n{END}"

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        if self.num_fewshot <= 0:
            return []
        if self.FEWSHOT_SPLIT not in self.dataset or not self.dataset[self.FEWSHOT_SPLIT]:
            return []
        available = self.dataset[self.FEWSHOT_SPLIT]
        n = min(self.num_fewshot, len(available))
        return self.rnd.sample(available, n)

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        """Extract code from between ```lang and ```."""
        text = completion_text.replace("\r\n", "\n")
        if BEGIN in text:
            parts = text.split(BEGIN, 1)
            if len(parts) > 1:
                after = parts[1]
                if "\n" in after:
                    _, code_part = after.split("\n", 1)
                else:
                    code_part = after
                if END in code_part:
                    code_part = code_part.split(END)[0]
                return code_part.strip()
        return text.strip()
