"""German Hendrycks Math (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/hendrycks-math-multilingual
"""

from typing import Any

from eval_framework.metrics.completion.math_minerva_completion import (
    MathMinervaCompletionDE,
    MathMinervaCompletionRelaxedDE,
)
from eval_framework.metrics.completion.minerva_math_utils import extract_answers
from eval_framework.metrics.efficiency.completion_tokens import NumCompletionTokens
from eval_framework.metrics.efficiency.reasoning_tokens import NumReasoningTokens
from eval_framework.tasks.base import Language, Sample
from eval_framework.tasks.benchmarks.math_reasoning import MATHMinerva
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle


class MATHMinervaDE_OLMES(MATHMinerva):
    """German Hendrycks Math - MATHMinerva-style reasoning generation.

    Dataset: https://huggingface.co/datasets/ellamind/hendrycks-math-multilingual

    Matches the English ``MATHMinerva_OLMES`` task with these German adaptations:
      * prompt markers translated: ``Aufgabe:`` / ``Lösung:``
      * stop sequences adjusted accordingly
      * few-shot examples drawn from the dataset (not the hardcoded English
        ``_OLMES_FEWSHOTS`` list the parent uses)
      * few-shot targets end with a German Minerva-style final-answer line.

    The EllaMind dataset is test-only; few-shot sampling uses the same
    oversample-and-drop guard as ``BaseTask._sample_fewshot_examples`` while
    filtering out multi-paragraph solutions. Solutions preserve the original
    ``\\boxed{...}`` markers, so the parent's ground-truth extraction and the
    Minerva metric's boxed-answer fallback both keep working unchanged.
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "MATHMinervaDE_OLMES"
    DATASET_PATH = "ellamind/hendrycks-math-multilingual"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"  # EllaMind only ships a test split
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    METRICS = [MathMinervaCompletionDE, MathMinervaCompletionRelaxedDE, NumCompletionTokens, NumReasoningTokens]

    def __init__(self, num_fewshot: int = 4) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["Aufgabe:", "\n\n"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return "Aufgabe:\n" + item["problem"] + "\n\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Lösung:"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        # Mirror the English Minerva format's "Final Answer: The final answer is ..."
        # so the model learns to terminate each solution with an explicit final-answer line.
        return (
            f"{self._get_cue_text(item)} {item['solution']}\n"
            f"Finale Antwort: Die finale Antwort lautet ${item['extracted_answer']}$. "
            "Ich hoffe, die Antwort ist korrekt."
        )

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        # Restrict to single-paragraph solutions so the "\n\n" stop sequence fires between
        # few-shot blocks rather than mid-solution
        pool = [ex for ex in self.dataset[self.FEWSHOT_SPLIT] if "\n\n" not in ex["solution"]]
        # Same-split guard: oversample by one and drop the target item if present.
        assert self.rnd is not None, "Task RNG is unseeded; build tasks via `with_overwrite`."
        candidates = self.rnd.sample(pool, self.num_fewshot + 1)
        return [ex for ex in candidates if ex != item][: self.num_fewshot]

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        candidates = extract_answers(completion_text, use_cot=True, cot_style="minerva_de", relaxed=True)
        return candidates[0] if candidates else "[no_answer]"


class MATHMinervaDE_BPB_OLMES(MATHMinervaDE_OLMES):
    """German HENDRYCKS Math - BPB format.

    Dataset: https://huggingface.co/datasets/ellamind/hendrycks-math-multilingual
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "MATHMinervaDE_BPB_OLMES"
    TASK_STYLER = BPBStyle(cue_text="")

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return item["problem"]

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return [
            (
                f"{item['solution']}\n"
                f"Finale Antwort: Die finale Antwort lautet ${item['extracted_answer']}$. "
                "Ich hoffe, die Antwort ist korrekt."
            )
        ]

    # Needs to be set again, because one of the parents overrides the default implementation.
    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        return f" {self._get_choices(item)[0]}"

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return 0


class MATHMinervaDE_OLMES_NONL(MATHMinervaDE_OLMES):
    """Same as MATHMinervaDE_OLMES, but without `\\n\\n` as stop sequence."""

    NAME = "MATHMinervaDE_OLMES_NONL"

    def __init__(self, num_fewshot: int = 4) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["Aufgabe:"]
