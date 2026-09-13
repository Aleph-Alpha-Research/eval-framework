"""German HumanEval (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/humaneval-multilingual
"""

from typing import Any

from eval_framework.tasks.base import BaseTask, Language, Sample
from eval_framework.tasks.benchmarks.humaneval import (
    CODE_TO_EXECUTE,
    HumanEval_OLMES,
    HumanEval_OLMES_V2,
    HumanEvalMetricContext,
)
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle
from eval_framework.tasks.utils import extract_python_code_from_response


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


class HumanEvalDE_BPB_OLMES_V2(HumanEvalDE_BPB_OLMES):
    """HumanEvalDE_BPB_OLMES variant that wraps the prompt and canonical solution in markdown code
    fences, mirroring the HumanEvalDE_OLMES completion prompt exactly.
    """

    NAME = "HumanEvalDE_BPB_OLMES V2"

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return [item["canonical_solution"].rstrip() + "\n```"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        # Ensure that the code completion starts on a new line.
        return "```python\n" + item["prompt"].rstrip() + "\n"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        return item["canonical_solution"].rstrip() + "\n```"


class HumanEvalDE_OLMES_V2(HumanEval_OLMES_V2):
    NAME = "Human Eval DE Newline OLMES"
    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    DATASET_PATH = "ellamind/humaneval-multilingual"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU


class HumanEvalDEInstruct(HumanEvalDE_OLMES_V2):
    """German HumanEval for instruction-tuned chat models.

    Zero-shot by default: the model is asked in natural language to return the completed function in a
    markdown code block, which is then extracted from the free-form response. Few-shot examples are
    supported but not required for chat models.
    """

    NAME = "Human Eval DE Instruct"

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = []

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return (
            "Vervollständige die folgende Python-Funktion. Gib ausschließlich die vollständige Funktion "
            f"in einem Markdown-Codeblock zurück:\n```python\n{item['prompt'].strip()}\n```\n"
        )

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        code = item["prompt"].strip() + "\n" + item["canonical_solution"].rstrip()
        return f"```python\n{code}\n```"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        assert sample is not None and isinstance(sample.context, HumanEvalMetricContext)
        return CODE_TO_EXECUTE.format(
            start_of_code="",
            completion_text=extract_python_code_from_response(completion_text),
            test_code=sample.context.test,
            entry_point=sample.context.entry_point,
        )
