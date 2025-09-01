import re
from abc import ABC
from typing import Any

from datasets import Features, Sequence, Value

from eval_framework.metrics.completion_metrics.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.loglikelihood_metrics.accuracy_loglikelihood import AccuracyLoglikelihood
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.dataloader import Dataloader


class InfiniteBench(BaseTask[str], ABC):
    """
    InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens
    https://github.com/OpenBMB/InfiniteBench
    """

    DATASET_PATH = "xinrongzhang2022/InfiniteBench"
    SUBJECTS = ["default"]
    LANGUAGE = Language.ENG
    PERTURBATION_UNMODIFIABLE_WORDS = None
    FEATURES = Features(
        {
            "id": Value("int64"),
            "context": Value("string"),
            "input": Value("string"),
            "answer": Sequence(Value("string")),
            "options": Sequence(Value("string")),
        }
    )

    def __init__(self, dataloader: Dataloader, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "Few-shots are not supported for long-context InfiniteBench tasks"
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)


class InfiniteBenchLoglikelihood(InfiniteBench, ABC):
    """Base class for loglikelihood tasks."""

    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"{item['context']}\n\n{item['input']}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        assert item["answer"][0] in item["options"], f"Ground truth {item['answer']} is not in {item['options']}"
        return f" {item['answer'][0]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {choice}" for choice in item["options"]]


class InfiniteBenchCompletion(InfiniteBench, ABC):
    """Base class for completion tasks."""

    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"{item['context']}\n\n{item['input']}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        return item["answer"]


class InfiniteBench_CodeDebug(InfiniteBenchLoglikelihood):
    """Finding which function in a code repo contains a crashing error (MC form)."""

    NAME = "InfiniteBench_CodeDebug"
    SAMPLE_SPLIT = "code_debug"
    FEWSHOT_SPLIT = SAMPLE_SPLIT


class InfiniteBench_EnMC(InfiniteBenchLoglikelihood):
    """Multiple choice questions derived from the fake book."""

    NAME = "InfiniteBench_EnMC"
    SAMPLE_SPLIT = "longbook_choice_eng"
    FEWSHOT_SPLIT = SAMPLE_SPLIT


class InfiniteBench_CodeRun(InfiniteBenchCompletion):
    """Simulating execution of multiple simple, synthetic functions."""

    NAME = "InfiniteBench_CodeRun"
    SAMPLE_SPLIT = "code_run"
    FEWSHOT_SPLIT = SAMPLE_SPLIT

    def __init__(self, dataloader: Dataloader, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)
        self.stop_sequences: list[str] = ["\n"]
        self.max_tokens = 30  # Avg Output Tokens: 1.3

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]

        ANS_RE = re.compile(r"The return value is: (\-?[0-9\.\,]+)")
        match = ANS_RE.search(completion_text)
        if match:
            match_str = match.group(1).strip()
            return match_str
        else:
            return "[invalid]"


class InfiniteBench_EnDia(InfiniteBenchCompletion):
    """Identification of talkers in partially anonymized scripts."""

    NAME = "InfiniteBench_EnDia"
    SAMPLE_SPLIT = "longdialogue_qa_eng"
    FEWSHOT_SPLIT = SAMPLE_SPLIT

    def __init__(self, dataloader: Dataloader, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)
        self.stop_sequences: list[str] = ["\n"]
        self.max_tokens = 30  # Avg Output Tokens: 3.4

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        answers = [i.lower() for i in item["answer"]]
        return answers

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"{item['context']}\n\n{item['input']}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "The character which is $$MASK$$ is:"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]
        return completion_text.lower()


class InfiniteBench_EnQA(InfiniteBenchCompletion):
    """Free-form question answering based on the fake book."""

    NAME = "InfiniteBench_EnQA"
    SAMPLE_SPLIT = "longbook_qa_eng"
    FEWSHOT_SPLIT = SAMPLE_SPLIT

    def __init__(self, dataloader: Dataloader, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)
        self.stop_sequences: list[str] = ["\n"]
        self.max_tokens = 30  # Avg Output Tokens: 4.8

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"{item['context']}\n{item['input']}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        answers = [i.replace('"', "").lower() for i in item["answer"]]
        return answers

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]
        return completion_text.lower()


class InfiniteBench_MathFind(InfiniteBenchCompletion):
    """Finding special integers in a lengthy list."""

    NAME = "InfiniteBench_MathFind"
    SAMPLE_SPLIT = "math_find"
    FEWSHOT_SPLIT = SAMPLE_SPLIT

    def __init__(self, dataloader: Dataloader, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)
        self.stop_sequences: list[str] = ["\n"]
        self.max_tokens = 30  # Avg Output Tokens: 1.3

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]

        ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
        match = ANS_RE.search(completion_text)
        if match:
            match_str = match.group(0).strip()
            return match_str
        else:
            return "[invalid]"


class InfiniteBench_RetrieveKV2(InfiniteBenchCompletion):
    """Finding the corresponding value from a dictionary and a key."""

    NAME = "InfiniteBench_RetrieveKV2"
    SAMPLE_SPLIT = "kv_retrieval"
    FEWSHOT_SPLIT = SAMPLE_SPLIT

    def __init__(self, dataloader: Dataloader, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)
        self.stop_sequences: list[str] = ["\n"]
        self.max_tokens = 40  # Avg Output Tokens: 22.7 (all answers are 36 chars)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"{item['context']}\n{item['input']}"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]

        ANS_RE = re.compile(r"([0-9a-f\-]+)")
        match = ANS_RE.search(completion_text)
        if match:
            match_str = match.group(1).strip()
            return match_str
        else:
            return "[invalid]"


class InfiniteBench_RetrieveNumber(InfiniteBenchCompletion):
    """Locating repeated hidden numbers in a noisy long context."""

    NAME = "InfiniteBench_RetrieveNumber"
    SAMPLE_SPLIT = "number_string"
    FEWSHOT_SPLIT = SAMPLE_SPLIT

    def __init__(self, dataloader: Dataloader, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)
        self.stop_sequences: list[str] = ["\n"]
        self.max_tokens = 12  # Avg Output Tokens: 4.0 (all answers are 10 digits integers)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"{item['context']}\n{item['input']}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "The sequence of digits is:"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]

        ANS_RE = re.compile(r"([0-9]+)")
        match = ANS_RE.search(completion_text)
        if match:
            match_str = match.group(1).strip()
            return match_str
        else:
            return "[invalid]"


class InfiniteBench_RetrievePassKey1(InfiniteBenchCompletion):
    """Retrieving hidden keys in a noisy long context."""

    NAME = "InfiniteBench_RetrievePassKey1"
    SAMPLE_SPLIT = "passkey"
    FEWSHOT_SPLIT = SAMPLE_SPLIT

    def __init__(self, dataloader: Dataloader, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)
        self.stop_sequences: list[str] = ["\n"]
        self.max_tokens = 8  # Avg Output Tokens: 2.0 (all answers are 5 digits integers)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"{item['context']}\n{item['input']}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "The pass key is:"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]

        ANS_RE = re.compile(r"([0-9]+)")
        match = ANS_RE.search(completion_text)
        if match:
            match_str = match.group(1).strip()
            return match_str
        else:
            return "[invalid]"
