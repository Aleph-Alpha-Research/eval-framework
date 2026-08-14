"""German HumanEval (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/humaneval-multilingual
"""

from typing import Any

from eval_framework.tasks.base import BaseTask, Language, Sample
from eval_framework.tasks.benchmarks.humaneval import HumanEval_NL, HumanEval_OLMES
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle


class HumanEvalDE_OLMES(HumanEval_OLMES):
    """German HumanEval - code generation format (pass@1 via test execution).

    Dataset: https://huggingface.co/datasets/ellamind/humaneval-multilingual

    The EllaMind dataset mirrors the original HumanEval dataset, so only the dataset path and
    language metadata are overridden.
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "HumanEvalDE_OLMES"
    DATASET_PATH = "ellamind/humaneval-multilingual"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU


class HumanEvalDE_BPB_OLMES(BaseTask[str]):
    """German HumanEval - BPB format (loglikelihood of the canonical solution).

    Dataset: https://huggingface.co/datasets/ellamind/humaneval-multilingual

    We use the following prompt format:
    {prompt}
    {canonical_solution}
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "HumanEvalDE_BPB_OLMES"
    DATASET_PATH = "ellamind/humaneval-multilingual"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    # No leading space: canonical solution is already indented by 4 spaces.
    TASK_STYLER = BPBStyle(question_prefix="", cue_text="", leading_space_continuations=False)

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return item["prompt"].strip("\n")

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return [item["canonical_solution"]]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return 0

    def _create_samples(self, item: dict[str, Any], index: int, subject: str) -> list[Sample]:
        """Creates one or more samples from a single dataset item. Default implementation returns single sample."""
        messages = self._get_messages(item)
        messages[-1].content = messages[-1].content.rstrip("\n")
        return [
            Sample(
                id=index,
                subject=str(subject),
                messages=messages,
                ground_truth=self._get_ground_truth(item),
                possible_completions=self._get_possible_completions(item),
                context=self._get_context(item),
            )
        ]


class HumanEvalDE_BPB_OLMES_V2(HumanEvalDE_BPB_OLMES):
    """HumanEvalDE_BPB_OLMES variant that wraps the prompt and canonical solution in markdown code
    fences, mirroring the HumanEvalDE_OLMES completion prompt exactly.
    """

    NAME = "HumanEvalDE_BPB_OLMES V2"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        # Ensure that the code completion starts on a new line.
        return "```python\n" + item["prompt"].rstrip()

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        return "\n" + item["canonical_solution"].rstrip() + "\n```"


class HumanEvalDE_NL(HumanEval_NL):
    NAME = "HumanEvalDE_NL"
    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    DATASET_PATH = "ellamind/humaneval-multilingual"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
