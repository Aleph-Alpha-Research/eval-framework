"""Task-format mixins and other helper functions related to task formatting.

This module provides two mixins — ``MCTaskMixin`` and ``ClozeTaskMixin`` that
reduce boilerplate in multiple-choice and cloze evaluation tasks. Using a mixin
is entirely optional; existing tasks that don't use them continue to work unchanged.

Quick-start
-----------
A new task only needs to implement **three data-access methods**:

* ``_get_raw_question(item) -> str``   — the bare question string (no prefix added)
* ``_get_choices(item) -> list[str]``  — ordered list of answer options
* ``_get_correct_index(item) -> int``  — 0-based index of the correct answer

Then mix in the desired format. Note, always list the mixin **before** the task base class so its
methods take precedence, e.g. for a Cloze task:

.. code-block:: python

    class MyTask(ClozeTaskMixin, BaseTask[str]):
        NAME = "MyTask"
        DATASET_PATH = "my/dataset"
        SAMPLE_SPLIT = "test"
        FEWSHOT_SPLIT = "train"
        SUBJECTS = ["my_subject"]
        LANGUAGE = Language.ENG
        PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]

        def _get_raw_question(self, item): return item["question"]
        def _get_choices(self, item): return item["choices"]
        def _get_correct_index(self, item): return item["answer_idx"]

For task families with both MC and Cloze variants, a non-registered base class
(named ``_XXX_Base`` by convention) holds the shared dataset attributes and the
three data-access methods. The registered variants are then thin variants:

.. code-block:: python

    class _ARC_Base(BaseTask[str]):
        DATASET_PATH = "allenai/ai2_arc"
        ...
        def _get_raw_question(self, item): return item["question"]
        def _get_choices(self, item): return item["choices"]["text"]
        def _get_correct_index(self, item): ...

    class ARC(ClozeTaskMixin, _ARC_Base):
        NAME = "ARC"

    class ARC_MC(MCTaskMixin, _ARC_Base):
        NAME = "ARC_MC"
        SPACE_PREFIXED_LABELS = True
"""

import hashlib
import random
from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.tasks.base import Language, ResponseType, TaskFormat
from eval_framework.tasks.utils import get_n_letters

# Default (QUESTION_PREFIX, CUE_TEXT) per Language if not set explicitly; extend for new languages as needed.
_DEFAULT_QUESTION_CUE_TEXT: dict[Language, tuple[str, str]] = {
    Language.ENG: ("Question: ", "Answer:"),
    Language.DEU: ("Frage: ", "Antwort:"),
}


class MCTaskMixin:
    """Multiple-choice format mixin: choices shown in prompt, model scored over letter labels.

    The mixin handles all prompt assembly and scoring logic automatically. The task
    class (or a shared base) only needs to implement three data-access methods:

    * ``_get_raw_question(item) -> str``
    * ``_get_choices(item) -> list[str]``
    * ``_get_correct_index(item) -> int``

    Class attributes to override on the task class
    -----------------------------------------------
    QUESTION_PREFIX (str):
        Prepended to the raw question to build the full question line.
        Default: ``"Question: "`` (or language-specific default).
    CUE_TEXT (str):
        Text appended as a cue message after the prompt.
        Default: ``"Answer:"`` (or language-specific default).
    SPACE_PREFIXED_LABELS (bool):
        When ``True`` each option line is indented with a space (``" A. choice"``).
        This is, e.g., the OLMES-style prompt variant. Default: ``False``.

    Assembled prompt example (default settings, 3 choices)::

        "Question: What is the capital of France?\\nA. Berlin\\nB. Paris\\nC. London\\n Answer:"

        Scored completions: [" A", " B", " C"]
        Ground truth:  " B"
    """

    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        BitsPerByteLoglikelihood,
    ]
    QUESTION_PREFIX: str = "Question: "
    CUE_TEXT: str = "Answer:"
    SPACE_PREFIXED_LABELS: bool = False
    TASK_FORMAT: TaskFormat = TaskFormat.MULTIPLE_CHOICE

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-fill QUESTION_PREFIX & CUE_TEXT from _DEFAULT_QUESTION_CUE_TEXT if not set explicitly."""
        super().__init_subclass__(**kwargs)
        lang = getattr(cls, "LANGUAGE", None)
        # If the language is not in _DEFAULT_QUESTION_CUE_TEXT, return early.
        if lang not in _DEFAULT_QUESTION_CUE_TEXT:
            return
        default_prefix, default_cue = _DEFAULT_QUESTION_CUE_TEXT[lang]
        # Walk the MRO (excluding this mixin) to find any explicit definition, otherwise set to default.
        if "QUESTION_PREFIX" not in cls.__dict__:
            cls.QUESTION_PREFIX = next(
                (
                    k.__dict__["QUESTION_PREFIX"]
                    for k in cls.__mro__
                    if k is not MCTaskMixin and "QUESTION_PREFIX" in k.__dict__
                ),
                default_prefix,
            )
        if "CUE_TEXT" not in cls.__dict__:
            cls.CUE_TEXT = next(
                (k.__dict__["CUE_TEXT"] for k in cls.__mro__ if k is not MCTaskMixin and "CUE_TEXT" in k.__dict__),
                default_cue,
            )

    def _get_question_text(self, item: dict[str, Any]) -> str:
        """Return the full question line (prefix + raw question).

        Override this method directly if the question format cannot be expressed
        as a simple prefix + raw-question string (e.g. HellaSwag's
        "activity: context" form).
        """
        return f"{self.QUESTION_PREFIX}{self._get_raw_question(item)}"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        """Return the full MC prompt: question line followed by labeled choices."""
        return format_mc_prompt(
            self._get_question_text(item),
            self._get_choices(item),
            space_prefixed_labels=self.SPACE_PREFIXED_LABELS,
        )

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        """Return the space-prefixed label letter of the correct answer (e.g. \" B\")."""
        labels = get_n_letters(len(self._get_choices(item)))
        return f" {labels[self._get_correct_index(item)]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        """Return all label letters as space-prefixed completions (e.g. [\" A\", \" B\", ...])."""
        return [f" {label}" for label in get_n_letters(len(self._get_choices(item)))]

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        """Return the assistant cue text (appended as an assistant-role message)."""
        return self.CUE_TEXT

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        """Return the few-shot target string, i.e. cue + ground truth (e.g. \"Answer: B\")."""
        gt = self._get_ground_truth(item)
        assert isinstance(gt, str)
        return f"{self._get_cue_text(item)}{gt}"

    def get_metadata(self) -> dict:
        """Add task_format to the task's metadata."""
        return {**super().get_metadata(), "task_format": self.TASK_FORMAT.value}  # type: ignore[misc]


class ClozeTaskMixin:
    """Cloze format mixin: no choices shown in prompt, model scored over full choice text.

    Also known as "ranked classification" (RC). The prompt only shows the question; the model's
    score for each full answer text (e.g. ``" Paris"``) determines the prediction.

    The same three data-access methods are required as for ``MCTaskMixin``:

    * ``_get_raw_question(item) -> str``
    * ``_get_choices(item) -> list[str]``
    * ``_get_correct_index(item) -> int``

    Class attributes to override on the task class
    -----------------------------------------------
    QUESTION_PREFIX, CUE_TEXT — same semantics as in ``MCTaskMixin``.
    TRAILING_NEWLINE (bool):
        When ``True`` (default), the instruction ends with ``"\\n"``.
        Set to ``False`` for sentence-completion tasks (HellaSwag, Winogrande)
        where the model should continue a sentence fragment directly.

    Assembled prompt example (3 choices)::

        "Question: What is the capital of France?\\nAnswer:"

        Scored completions: [" Berlin", " Paris", " London"]
        Ground truth: " Paris"

    Sentence-completion example (TRAILING_NEWLINE=False, CUE_TEXT="")::

        "The cat sat on the"

        Scored completions: [" mat", " floor", " sofa"]
        Ground truth: " mat"
    """

    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        BitsPerByteLoglikelihood,
    ]
    QUESTION_PREFIX: str = "Question: "
    CUE_TEXT: str = "Answer:"
    TRAILING_NEWLINE: bool = True
    TASK_FORMAT: TaskFormat = TaskFormat.CLOZE

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-fill QUESTION_PREFIX & CUE_TEXT from _DEFAULT_QUESTION_CUE_TEXT if not set explicitly."""
        super().__init_subclass__(**kwargs)
        lang = getattr(cls, "LANGUAGE", None)
        # If the language is not in _DEFAULT_QUESTION_CUE_TEXT, return early.
        if lang not in _DEFAULT_QUESTION_CUE_TEXT:
            return
        default_prefix, default_cue = _DEFAULT_QUESTION_CUE_TEXT[lang]
        # Walk the MRO (excluding this mixin) to find any explicit definition, otherwise use default.
        if "QUESTION_PREFIX" not in cls.__dict__:
            cls.QUESTION_PREFIX = next(
                (
                    k.__dict__["QUESTION_PREFIX"]
                    for k in cls.__mro__
                    if k is not ClozeTaskMixin and "QUESTION_PREFIX" in k.__dict__
                ),
                default_prefix,
            )
        if "CUE_TEXT" not in cls.__dict__:
            cls.CUE_TEXT = next(
                (k.__dict__["CUE_TEXT"] for k in cls.__mro__ if k is not ClozeTaskMixin and "CUE_TEXT" in k.__dict__),
                default_cue,
            )

    def _get_question_text(self, item: dict[str, Any]) -> str:
        """Return the full question line (prefix + raw question).

        Override this method directly if the question format cannot be expressed
        as a simple prefix + raw-question string.
        """
        return f"{self.QUESTION_PREFIX}{self._get_raw_question(item)}"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        """Return the question line, optionally with a trailing newline."""
        text = self._get_question_text(item)
        return f"{text}\n" if self.TRAILING_NEWLINE else text

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        """Return the space-prefixed text of the correct choice (e.g. \" Paris\")."""
        choices = self._get_choices(item)
        return f" {choices[self._get_correct_index(item)]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        """Return all choice texts as space-prefixed completions (e.g. [\" Berlin\", ...])."""
        return [f" {c}" for c in self._get_choices(item)]

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        """Return the assistant cue text (appended as an assistant-role message)."""
        return self.CUE_TEXT

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        """Return the few-shot target string, i.e. cue + ground truth (e.g. \"Answer: Paris\")."""
        gt = self._get_ground_truth(item)
        assert isinstance(gt, str)
        return f"{self._get_cue_text(item)}{gt}"

    def get_metadata(self) -> dict:
        """Extend base metadata with the task format."""
        return {**super().get_metadata(), "task_format": self.TASK_FORMAT.value}  # type: ignore[misc]


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
