"""German Winogrande (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/winogrande-multilingual

The sentence contains a blank ``_`` to be filled by option1 or option2.
All three styles (Cloze, MC, PartialEval) share this extraction logic.
"""

from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import PartialEvalAccuracy
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import ClozeStyle, MCStyle
from template_formatting.formatter import Message, Role

_WINOGRANDE_ANSWER_STR_TO_NUM = {"1": 0, "2": 1}


class _WINOGRANDE_ELLAMIND_DE_Base(BaseTask[str]):
    """Non-registered base for German Winogrande (EllaMind) variants.

    Dataset: https://huggingface.co/datasets/ellamind/winogrande-multilingual

    The sentence contains a blank ``_`` to be filled by option1 or option2.
    All three styles (Cloze, MC, PartialEval) share this extraction logic.
    """

    DATASET_PATH = "ellamind/winogrande-multilingual"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    PERTURBATION_UNMODIFIABLE_WORDS = ["1", "2"]

    def _extract_prefix(self, item: dict[str, Any]) -> str:
        """Return the sentence fragment before the blank (stripped)."""
        prefix, _ = item["sentence"].split("_", 1)
        return prefix.replace("  ", " ").strip()

    def _extract_suffix(self, item: dict[str, Any]) -> str:
        """Return the sentence fragment after the blank (leading space preserved)."""
        _, suffix = item["sentence"].split("_", 1)
        return suffix.replace("  ", " ")

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return self._extract_prefix(item)

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        suffix = self._extract_suffix(item)
        return [item["option1"] + suffix, item["option2"] + suffix]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return _WINOGRANDE_ANSWER_STR_TO_NUM[item["answer"]]


class WINOGRANDE_ELLAMIND_CLOZE_DE(_WINOGRANDE_ELLAMIND_DE_Base):
    """German Winogrande - Cloze format (like WINOGRANDE).

    Prompt is the sentence prefix; model is scored over the full
    "option + suffix" strings as completions.

    Hypothetical example:
        Prompt:      "George was taller than John, therefore"
        Completions: [" George could see more.", " John could see more."]
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "WINOGRANDE_ELLAMIND_CLOZE_DE"
    TASK_STYLER = ClozeStyle(question_prefix="", trailing_newline=False, cue_text="")


class WINOGRANDE_ELLAMIND_MC_DE(_WINOGRANDE_ELLAMIND_DE_Base):
    """German Winogrande - MC format (like WINOGRANDE_OLMES).

    Both options are listed in the prompt with space-prefixed letter labels
    (" A.", " B."); model is scored over " A" / " B".

    Hypothetical example:
        Prompt:      "George was taller than John, therefore\\n A. George could see more.\\n B. John could see more.\\n"
        Completions: [" A", " B"]
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "WINOGRANDE_ELLAMIND_MC_DE"
    TASK_STYLER = MCStyle().for_language(Language.DEU)


class WINOGRANDE_ELLAMIND_PARTIAL_EVAL_DE(_WINOGRANDE_ELLAMIND_DE_Base):
    """German Winogrande - PartialEval cloze format (like WINOGRANDECloze).

    Evaluates p(suffix | prefix + option) rather than p(option + suffix | prefix).

    Two samples are created per dataset item - one per option - each with prompt
    "prefix + option" and the shared suffix as the single possible completion.
    Uses PartialEvalAccuracy, which pairs the two samples and compares scores.

    Hypothetical example (two samples for one item):
        Prompt 1: "George was taller than John, therefore George"  completion: [" could see more."]
        Prompt 2: "George was taller than John, therefore John"    completion: [" could see more."]
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "WINOGRANDE_ELLAMIND_PARTIAL_EVAL_DE"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [PartialEvalAccuracy]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        # Used for few-shot examples: plain cloze prefix as prompt.
        return self._extract_prefix(item)

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        # Used for few-shot examples: correct "option + suffix" with leading space.
        choices = self._get_choices(item)
        return f" {choices[self._get_correct_index(item)]}"

    def _create_samples(self, item: dict[str, Any], index: int, subject: str) -> list[Sample]:
        """Create two samples per item, one for each option."""
        fewshot_messages = self._get_example_messages(item)
        # Keep trailing space in prefix so "prefix + option" is well-formed.
        prefix, _ = item["sentence"].split("_", 1)
        prefix = prefix.replace("  ", " ")
        suffix = self._extract_suffix(item)
        ground_truth_option = _WINOGRANDE_ANSWER_STR_TO_NUM[item["answer"]]

        samples = []
        for opt_idx, option in enumerate([item["option1"], item["option2"]]):
            messages = fewshot_messages + [Message(role=Role.USER, content=f"{prefix}{option}")]
            samples.append(
                Sample(
                    id=index + opt_idx,
                    subject=str(subject),
                    messages=messages,
                    ground_truth=str(opt_idx == ground_truth_option),
                    possible_completions=[suffix],
                )
            )
        return samples
