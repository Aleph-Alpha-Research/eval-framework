"""German HellaSwag (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/hellaswag-multilingual

HellaSwag supplies separate easy and hard distractors. Each task class uses a
``_DISTRACTOR_LEVEL`` class attribute (``"easy"`` or ``"hard"``).
"""

from typing import Any, Literal

from eval_framework.tasks.base import BaseTask, Language
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, shuffle_correct_with_distractors


class HELLASWAG_ELLAMIND_EASY_DE(BaseTask[str]):
    """German HellaSwag - Cloze (sentence-completion) format with easy distractors.

    Dataset: https://huggingface.co/datasets/ellamind/hellaswag-multilingual

    HellaSwag is a *sentence-completion* task: the prompt is a partial sentence
    (``"{activity}: {context}"``) and the model scores full sentence endings.
    There is no natural MC variant for this task (would be possible, but not natural).

    Set ``_DISTRACTOR_LEVEL = "easy"`` or ``"hard"`` on the task class.
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "HELLASWAG_ELLAMIND_EASY_DE"
    DATASET_PATH = "ellamind/hellaswag-multilingual"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    _DISTRACTOR_LEVEL: Literal["easy", "hard"] = "easy"
    # Sentence-completion: no question prefix, no cue, continuation follows directly
    TASK_STYLER = ClozeStyle(question_prefix="", trailing_newline=False, cue_text="")

    def _shuffled(self, item: dict[str, Any]) -> tuple[list[str], int]:
        distractors = item["easy_distractors"] if self._DISTRACTOR_LEVEL == "easy" else item["hard_distractors"]
        return shuffle_correct_with_distractors(
            correct=item["correct_ending"],
            distractors=distractors,
            seed_text=item["context"] + item["correct_ending"],
        )

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return f"{item['activity'].strip()}: {item['context'].strip()}"

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return self._shuffled(item)[0]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return self._shuffled(item)[1]


class HELLASWAG_ELLAMIND_HARD_DE(HELLASWAG_ELLAMIND_EASY_DE):
    """German HellaSwag - Cloze (sentence-completion) format with hard distractors."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "HELLASWAG_ELLAMIND_HARD_DE"
    _DISTRACTOR_LEVEL = "hard"


class HELLASWAG_ELLAMIND_BPB_DE(HELLASWAG_ELLAMIND_EASY_DE):
    """German HellaSwag - BPB format."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "HELLASWAG_ELLAMIND_BPB_DE"
    TASK_STYLER = BPBStyle(question_prefix="", trailing_newline=False, cue_text="")
