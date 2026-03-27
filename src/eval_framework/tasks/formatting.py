"""Task-format formatters and helper functions for choice-based evaluation tasks.

This module provides injectable formatting strategies — ``MCFormatter`` and
``ClozeFormatter`` — that reduce boilerplate in multiple-choice and cloze
evaluation tasks.  Using a formatter is entirely optional; existing tasks that
override ``BaseTask`` methods directly continue to work unchanged.

Quick-start
-----------
A new choice-based task sets ``TASK_STYLER`` on its ``BaseTask`` subclass and
implements **three data-access methods**:

* ``_get_raw_question(item) -> str``   — the bare question string
* ``_get_choices(item) -> list[str]``  — ordered list of answer options
* ``_get_correct_index(item) -> int``  — 0-based index of the correct answer

``BaseTask`` automatically delegates its formatting hooks to the formatter.
``RESPONSE_TYPE`` and ``METRICS`` are also auto-populated from the formatter.

.. code-block:: python

    class MyTask(BaseTask[str]):
        NAME = "MyTask"
        DATASET_PATH = "my/dataset"
        SAMPLE_SPLIT = "test"
        FEWSHOT_SPLIT = "train"
        SUBJECTS = ["my_subject"]
        PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
        TASK_STYLER = ClozeFormatter(question_prefix="Question: ", cue_text="Answer:")

        def _get_raw_question(self, item): return item["question"]
        def _get_choices(self, item): return item["choices"]
        def _get_correct_index(self, item): return item["answer_idx"]

For task families with both MC and Cloze variants, a shared base class holds the
dataset attributes and data-access methods.  Variants only differ in ``TASK_STYLER``:

.. code-block:: python

    class _ARC_Base(BaseTask[str]):
        DATASET_PATH = "allenai/ai2_arc"
        ...
        def _get_raw_question(self, item): return item["question"]
        def _get_choices(self, item): return item["choices"]["text"]
        def _get_correct_index(self, item): ...

    class ARC(_ARC_Base):
        NAME = "ARC"
        TASK_STYLER = ClozeFormatter()

    class ARC_MC(_ARC_Base):
        NAME = "ARC_MC"
        TASK_STYLER = MCFormatter(space_prefixed_labels=True)
"""

import hashlib
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.tasks.base import Language, ResponseType, TaskFormat
from eval_framework.tasks.utils import get_n_letters

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric

# Default (question_prefix, cue_text) per language; extend for new languages as needed.
_DEFAULT_QUESTION_CUE_TEXT: dict[Language, tuple[str, str]] = {
    Language.ENG: ("Question: ", "Answer:"),
    Language.DEU: ("Frage: ", "Antwort:"),
}


# ---------------------------------------------------------------------------
# Formatter strategy classes
# ---------------------------------------------------------------------------


class TaskFormatter(ABC):
    """Strategy object that controls prompt assembly and scoring for choice-based tasks.

    Concrete implementations (``MCFormatter``, ``ClozeFormatter``) are assigned to a
    task's ``TASK_STYLER`` class attribute.  ``BaseTask`` delegates its
    formatting hooks to this object, so task authors only implement data-access
    methods.

    Attributes:
        response_type: The response type the task should use (e.g. LOGLIKELIHOODS).
        metrics:       Default metric classes for tasks using this formatter.
        task_format:   Discriminator for metadata (MULTIPLE_CHOICE or CLOZE).
        question_prefix: String prepended to the raw question.
    """

    response_type: ResponseType
    metrics: list[type["BaseMetric"]]
    task_format: TaskFormat
    question_prefix: str

    @abstractmethod
    def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
        """Build the instruction/prompt text from a question and answer choices."""

    @abstractmethod
    def get_ground_truth(self, choices: list[str], correct_index: int) -> str:
        """Return the ground-truth string for scoring."""

    @abstractmethod
    def get_possible_completions(self, choices: list[str]) -> list[str]:
        """Return the list of scored completion strings."""

    @abstractmethod
    def get_cue_text(self) -> str:
        """Return the assistant cue appended after the prompt (e.g. ``"Answer:"``)."""

    def get_question_text(self, raw_question: str) -> str:
        """Build the full question line (prefix + raw question).

        Override in a subclass for non-standard question formats (e.g. HellaSwag's
        ``"activity: context"`` form).
        """
        return f"{self.question_prefix}{raw_question}"

    def get_fewshot_target_text(self, choices: list[str], correct_index: int) -> str:
        """Return the few-shot target string (cue + ground truth)."""
        return f"{self.get_cue_text()}{self.get_ground_truth(choices, correct_index)}"

    def get_extra_metadata(self) -> dict:
        """Return formatter-specific metadata to merge into the task's metadata."""
        return {"task_format": self.task_format.value}

    @classmethod
    def for_language(cls, language: Language, **kwargs: Any) -> Self:
        """Factory that fills ``question_prefix`` and ``cue_text`` from language defaults.

        Any explicitly passed keyword arguments take precedence over the defaults.
        """
        if language in _DEFAULT_QUESTION_CUE_TEXT:
            prefix, cue = _DEFAULT_QUESTION_CUE_TEXT[language]
            kwargs.setdefault("question_prefix", prefix)
            kwargs.setdefault("cue_text", cue)
        return cls(**kwargs)


class MCFormatter(TaskFormatter):
    """Multiple-choice formatter: choices shown in prompt, model scored over letter labels.

    Args:
        question_prefix:        Prepended to the raw question (default ``"Question: "``).
        cue_text:               Assistant cue after the prompt (default ``"Answer:"``).
        space_prefixed_labels:  When ``True``, each option line starts with a space
                                (``" A. choice"`` — OLMES-style). Default ``False``.

    Assembled prompt example (default settings, 3 choices)::

        "Question: What is the capital of France?\\nA. Berlin\\nB. Paris\\nC. London\\n"

        Scored completions: [" A", " B", " C"]
        Ground truth:  " B"
    """

    response_type = ResponseType.LOGLIKELIHOODS
    metrics: list[type["BaseMetric"]] = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        BitsPerByteLoglikelihood,
    ]
    task_format = TaskFormat.MULTIPLE_CHOICE

    def __init__(
        self,
        question_prefix: str = "Question: ",
        cue_text: str = "Answer:",
        space_prefixed_labels: bool = False,
    ) -> None:
        self.question_prefix = question_prefix
        self._cue_text = cue_text
        self.space_prefixed_labels = space_prefixed_labels

    def get_cue_text(self) -> str:
        return self._cue_text

    def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
        return format_mc_prompt(
            self.get_question_text(raw_question),
            choices,
            space_prefixed_labels=self.space_prefixed_labels,
        )

    def get_ground_truth(self, choices: list[str], correct_index: int) -> str:
        labels = get_n_letters(len(choices))
        return f" {labels[correct_index]}"

    def get_possible_completions(self, choices: list[str]) -> list[str]:
        return [f" {label}" for label in get_n_letters(len(choices))]


class ClozeFormatter(TaskFormatter):
    """Cloze formatter: no choices in prompt, model scored over full choice text.

    Also known as "ranked classification" (RC).  The prompt only shows the question;
    the model's score for each full answer text determines the prediction.

    Args:
        question_prefix:   Prepended to the raw question (default ``"Question: "``).
        cue_text:          Assistant cue after the prompt (default ``"Answer:"``).
        trailing_newline:  When ``True`` (default), the instruction ends with ``"\\n"``.
                           Set to ``False`` for sentence-completion tasks where the
                           model should continue a fragment directly.

    Assembled prompt example (3 choices)::

        "Question: What is the capital of France?\\n"

        Scored completions: [" Berlin", " Paris", " London"]
        Ground truth: " Paris"

    Sentence-completion example (trailing_newline=False, cue_text="")::

        "The cat sat on the"

        Scored completions: [" mat", " floor", " sofa"]
        Ground truth: " mat"
    """

    response_type = ResponseType.LOGLIKELIHOODS
    metrics: list[type["BaseMetric"]] = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        BitsPerByteLoglikelihood,
    ]
    task_format = TaskFormat.CLOZE

    def __init__(
        self,
        question_prefix: str = "Question: ",
        cue_text: str = "Answer:",
        trailing_newline: bool = True,
    ) -> None:
        self.question_prefix = question_prefix
        self._cue_text = cue_text
        self.trailing_newline = trailing_newline

    def get_cue_text(self) -> str:
        return self._cue_text

    def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
        text = self.get_question_text(raw_question)
        return f"{text}\n" if self.trailing_newline else text

    def get_ground_truth(self, choices: list[str], correct_index: int) -> str:
        return f" {choices[correct_index]}"

    def get_possible_completions(self, choices: list[str]) -> list[str]:
        return [f" {c}" for c in choices]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def shuffle_correct_with_distractors(
    correct: str,
    distractors: list[str],
    seed_text: str,
) -> tuple[list[str], int]:
    """Deterministically shuffle distractors + correct answer; return (choices, correct_index).

    Calling this multiple times with the same arguments always returns the same result.

    Args:
        correct:     The correct answer string.
        distractors: The distractor strings, e.g. wrong answer choices.
        seed_text:   Text used as the shuffle seed (typically question + answer).

    Returns:
        A tuple ``(shuffled_choices, correct_index)`` where ``correct_index``
        is the 0-based position of ``correct`` in ``shuffled_choices``.
    """
    choices = [*distractors, correct]
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    order = list(range(len(choices)))
    rng.shuffle(order)
    shuffled = [choices[i] for i in order]
    correct_index = order.index(len(choices) - 1)
    return shuffled, correct_index


def answer_key_to_index(key: str) -> int:
    """Convert a letter or 1-based integer answer key to a 0-based index.

    Datasets sometimes encode the correct answer as a letter ("A", "B", ...)
    and sometimes as a 1-based integer string ("1", "2", ...).  This function
    normalises both to a 0-based index so task code doesn't need to branch.

    Args:
        key: A single-character string. Either a letter ("A"-"Z") or a digit ("1"-"9").

    Returns:
        0-based index: "A" or "1" → 0, "B" or "2" → 1, etc.
    """
    if key.isdigit():
        return int(key) - 1  # Shift 1-based integer by 1.
    return ord(key.upper()) - ord("A")  # Turn letter to 0-based index.


def format_mc_prompt(
    question_text: str,
    choices: list[str],
    *,
    space_prefixed_labels: bool = False,
) -> str:
    """Helper function to format a question and its labeled choices into a multiple-choice prompt.

    The choices are labeled A, B, C, ... in order and ends with a newline.

    Args:
        question_text:        The full question string (prefix already included).
        choices:              Ordered list of answer option strings.
        space_prefixed_labels: When ``True``, each option line is prefixed with a
                              space — " A. choice" instead of "A. choice".
                              This is, e.g., the OLMES-style prompt format.

    Returns:
        A string of the form ``"<question_text>\\n[pfx]A. choice0\\n[pfx]B. choice1\\n"``.

    Examples::

        >>> format_mc_prompt("Question: What is 1+1?", ["1", "2", "3"])
        'Question: What is 1+1?\\nA. 1\\nB. 2\\nC. 3\\n'
        >>> format_mc_prompt("Question: What is 1+1?", ["1", "2"], space_prefixed_labels=True)
        'Question: What is 1+1?\\n A. 1\\n B. 2\\n'
    """
    labels = get_n_letters(len(choices))
    pfx = " " if space_prefixed_labels else ""
    options = "\n".join(f"{pfx}{label}. {choice}" for label, choice in zip(labels, choices))
    return f"{question_text}\n{options}\n"
