"""German MBPP (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/mbpp-multilingual
"""

from typing import Any

from eval_framework.tasks.base import BaseTask, Language
from eval_framework.tasks.benchmarks.mbpp import MBPP_OLMES, MBPP_BPB_EvalPlus, MBPP_EvalPlus
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle


class MBPPDE_OLMES(MBPP_OLMES):
    """German MBPP - code generation format (pass@1 via test execution).

    Dataset: https://huggingface.co/datasets/ellamind/mbpp-multilingual

    The EllaMind dataset mirrors the google-research-datasets/mbpp schema
    (``text`` / ``code`` / ``test_list``), so the upstream MBPP stop sequence,
    cue text, metric context, and post-processing (which expands the generated
    code with the ground-truth asserts and prints ``True``/``False``) are
    reused unchanged. Only the dataset path, language metadata, and the German
    instruction wrapper around ``item["text"]`` are overridden.
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "MBPPDE_OLMES"
    DATASET_PATH = "ellamind/mbpp-multilingual"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"  # EllaMind only ships a test split
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    PERTURBATION_UNMODIFIABLE_WORDS = ["Aufgabe", "Lösung", "Loesung", "Losung"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        text = item["text"] if "text" in item else item["prompt"]
        test = item["test_list"][0]
        return (
            "Bitte erstelle ein in sich geschlossenes Python-Skript, "
            f"das das folgende Problem in einem Markdown-Code-Block löst:\n```\n{text.strip()}\n{test}\n```\n"
        )

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Hier ist die fertige Funktion:\n\n```python\n"

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        # Use the regular same-split sampling logic, instead of MBPP_OLMES's hard-coded English few-shots.
        return BaseTask._sample_fewshot_examples(self, item)


class MBPPDE_BPB_OLMES(BaseTask[str]):
    """German MBPP - BPB format (loglikelihood of the reference Python solution).

    Dataset: https://huggingface.co/datasets/ellamind/mbpp-multilingual

    We use the following prompt format:
    {text}
    ```python
    {code}
    ```
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "MBPPDE_BPB_OLMES"
    DATASET_PATH = "ellamind/mbpp-multilingual"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    PERTURBATION_UNMODIFIABLE_WORDS = None
    TASK_STYLER = BPBStyle(
        question_prefix="",
        trailing_newline=False,
        cue_text="",
        leading_space_continuations=False,
    )

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return item["text"].replace("\r\n", "\n").strip() + "\n```python\n"

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        code = item["code"] or ""
        return [code.replace("\r\n", "\n").strip() + "\n```"]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return 0


class MBPPDE_EvalPlus(MBPP_EvalPlus):
    """German MBPP - EvalPlus code-generation format (pass@1 via test execution).

    German pendant of MBPP_EvalPlus: same prompt structure and post-processing, with the German
    instruction wrapper, German response preamble, and same-split (German) few-shot sampling.
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "MBPPDE_EvalPlus"
    DATASET_PATH = "ellamind/mbpp-multilingual"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU

    def __init__(self, num_fewshot: int = 3) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["```", "\nassert", "\nBitte erstelle"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        text = item["text"] if "text" in item else item["prompt"]
        text = text.replace("\r\n", "\n").strip()
        test = item["test_list"][0]
        return (
            "Bitte erstelle ein in sich geschlossenes Python-Skript, "
            f"das das folgende Problem in einem Markdown-Code-Block löst:\n```\n{text}\n{test}\n```\n"
        )

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        code = item["code"].replace("\r\n", "\n").strip()
        return self._get_cue_text(item) + "\n" + code + "\n```"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return (
            "Hier ist ein Python-Skript mit einer in sich geschlossenen Funktion, "
            "die das Problem löst und die entsprechenden Tests besteht:\n```python"
        )

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        # Use the regular same-split sampling logic, instead of MBPP_EvalPlus's hard-coded English few-shots.
        return BaseTask._sample_fewshot_examples(self, item)


class MBPPDE_BPB_EvalPlus(MBPP_BPB_EvalPlus):
    """German MBPP - EvalPlus BPB format (loglikelihood of the reference Python solution).

    German pendant of MBPP_BPB_EvalPlus: same BPBStyle mechanism and EvalPlus prompt, with the German
    instruction wrapper, German response preamble, and same-split (German) few-shot sampling.
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "MBPPDE_BPB_EvalPlus"
    DATASET_PATH = "ellamind/mbpp-multilingual"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    PERTURBATION_UNMODIFIABLE_WORDS = None
    TASK_STYLER = BPBStyle(
        question_prefix="",
        trailing_newline=False,
        cue_text=(
            "Hier ist ein Python-Skript mit einer in sich geschlossenen Funktion, "
            "die das Problem löst und die entsprechenden Tests besteht:\n```python"
        ),
        leading_space_continuations=False,
    )

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        text = item["text"] if "text" in item else item["prompt"]
        text = text.replace("\r\n", "\n").strip()
        test = item["test_list"][0]
        return (
            "Bitte erstelle ein in sich geschlossenes Python-Skript, "
            f"das das folgende Problem in einem Markdown-Code-Block löst:\n```\n{text}\n{test}\n```\n"
        )

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        code = item["code"] or ""
        return ["\n" + code.replace("\r\n", "\n").strip() + "\n```"]

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        # Use the regular same-split sampling logic, instead of MBPP_BPB_EvalPlus's hard-coded English few-shots.
        return BaseTask._sample_fewshot_examples(self, item)
