"""German HumanEval (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/humaneval-multilingual
"""

from typing import Any

from eval_framework.tasks.base import BaseTask, Language
from eval_framework.tasks.benchmarks.humaneval import HumanEval_OLMES
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

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        # Ensure that the code completion starts on a new line.
        return "```python\n" + item["prompt"].rstrip() + "\n"


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
