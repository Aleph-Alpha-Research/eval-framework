import logging
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar

import iso639
from datasets import DatasetDict, DownloadConfig, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError
from pydantic import BaseModel, ConfigDict

from eval_framework.shared.types import BaseMetricContext
from template_formatting.formatter import Message, Role

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric

RANDOM_SEED = 42
NO_SUBJECT = "no_subject"


class ResponseType(Enum):
    COMPLETION = "completion"
    LOGLIKELIHOODS = "loglikelihoods"


class Language(Enum):
    ENG = "English"
    DEU = "German"
    FRA = "French"
    ITA = "Italian"
    SPA = "Spanish"
    POR = "Portuguese"
    NLD = "Dutch"
    FIN = "Finnish"
    SWE = "Swedish"
    ARB = "Arabic"
    POL = "Polish"
    RUS = "Russian"
    UKR = "Ukrainian"
    HRV = "Croatian"
    SRP = "Serbian"

    @classmethod
    def add_members(cls, new_members: dict[str, Any]) -> type["Language"]:
        members = {member.name: member.value for member in cls}
        for name, value in new_members.items():
            if name not in members:
                members[name] = value
        return Enum(cls.__name__, members)  # type: ignore[return-value]


languages: dict[str, str] = {}
for language in iso639.ALL_LANGUAGES:
    enum_name = language.part3.upper()
    languages[enum_name] = language.name

Language: type[Enum] = Language.add_members(languages)  # type: ignore[no-redef]


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    subject: str
    messages: list[Message]
    ground_truth: str | list[str] | None
    possible_completions: list[str] | None
    context: BaseMetricContext | list[BaseMetricContext] | None = None


SubjectType = TypeVar("SubjectType")

logger = logging.getLogger(__name__)


class BaseTask[SubjectType](ABC):
    NAME: str
    DATASET_PATH: str
    SAMPLE_SPLIT: str
    FEWSHOT_SPLIT: str
    RESPONSE_TYPE: ResponseType
    METRICS: list[type["BaseMetric"]]
    SUBJECTS: list[SubjectType]
    HF_REVISION: str | None = None  # tag name, or branch name, or commit hash to ensure reproducibility

    # Words in _get_instruction_text() not to be perturbed. List of words is case insensitive. No special characters
    # or whitespace should be included.
    PERTURBATION_UNMODIFIABLE_WORDS: list[str] | None

    # The language (or languages) tested by the benchmark. Accepts a single string, a dictionary specifying
    # language by subtopic, or `None` (for tasks not specific to a single language).
    LANGUAGE: Language | dict[str, Language] | dict[str, tuple[Language, Language]] | None

    def __init__(self, num_fewshot: int = 0) -> None:
        self.num_fewshot = num_fewshot
        self.stop_sequences: list[str] | None = None
        self.max_tokens: int | None = None

    @classmethod
    def with_overwrite(
        cls, num_fewshot: int, *, custom_subjects: list[str] | None, custom_hf_revision: str | None
    ) -> Self:
        instance = cls(num_fewshot=num_fewshot)

        # If custom subjects were provided during initialization, they take precedence over the class-level SUBJECTS.
        filtered_subjects = instance._filter_task_subjects(custom_subjects=custom_subjects)
        if filtered_subjects:
            logger.info(f"Setting SUBJECTS to `{filtered_subjects}` for the task {instance.__class__.__name__}")
            instance.SUBJECTS = filtered_subjects  # type: ignore[assignment]

        # If a custom revision was provided during initialization, it takes precedence over the class-level HF_REVISION.
        if custom_hf_revision:
            logger.info(f"Setting HF revision to `{custom_hf_revision}` for the task {instance.__class__.__name__}")
            instance.HF_REVISION = custom_hf_revision

        return instance

    def _filter_task_subjects(self, custom_subjects: list[str] | None) -> list[str] | list[tuple] | None:
        """Process custom subjects passed from EvalConfig. Check and returns restricted task subjects if specified."""
        if not custom_subjects:
            return None

        assert hasattr(self, "SUBJECTS") and len(self.SUBJECTS) > 0
        if isinstance(self.SUBJECTS[0], tuple):
            # subjects are specified as strings but we need tuples
            filters = [tuple(item.strip() for item in subject.split(",")) for subject in custom_subjects]

            # check if all parts of custom subjects exists (* is a wildcard)
            num_items = len(self.SUBJECTS[0])
            legal_values = [
                set([s[i] for s in self.SUBJECTS if isinstance(s, tuple)] + ["*"]) for i in range(num_items)
            ]

            for tpl in filters:
                for i, v in enumerate(tpl):
                    assert v in legal_values[i], f"Subject part {v} not found in task {self.__class__.__name__}"

            # filter task subjects. * is a supported wildcard for a specific item in a tuple, e.g. "DE_DE, *"
            chosen_subjects = []
            for subject in self.SUBJECTS:
                subject_tuple = subject if isinstance(subject, tuple) else tuple(str(subject).split(","))
                for filter in filters:
                    if all(filter[i] == "*" or filter[i] == subject_tuple[i] for i in range(num_items)):
                        chosen_subjects.append(subject_tuple)
                        break
            return chosen_subjects  # type: ignore[return-value]
        else:
            for cs in custom_subjects:
                assert cs in self.SUBJECTS, f"Subject {cs} not found in task {self.__class__.__name__}"
            return custom_subjects  # type: ignore[return-value]

    def _load_hf_dataset(self, **kwargs: Any) -> Any:
        # Check if the HF_REVISION is valid before loading the dataset
        if self.HF_REVISION:
            try:
                _ = HfApi().dataset_info(repo_id=kwargs["path"], revision=self.HF_REVISION, timeout=100.0)
            except Exception as e:
                if isinstance(e, RevisionNotFoundError):
                    raise e

        cache_dir: str = os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets")
        download_config = DownloadConfig(cache_dir=cache_dir, max_retries=5)
        try:
            return load_dataset(
                **kwargs,
                revision=self.HF_REVISION,
                trust_remote_code=True,
                cache_dir=cache_dir,
                download_config=download_config,
            )
        except Exception:
            return load_dataset(
                **kwargs,
                revision=self.HF_REVISION,
                trust_remote_code=True,
                cache_dir=f"{Path.home()}/.cache/eval-framework",
            )

    def _shuffle_splits(self, hf_dataset: DatasetDict) -> dict[str, Any]:
        dataset = {}
        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            if split not in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                continue

            data_list = list(data)

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            dataset[split] = data_list

        return dataset

    def _load_dataset(self, subject: SubjectType) -> None:
        name = subject if subject != NO_SUBJECT else None
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name=name)
        self.dataset = self._shuffle_splits(hf_dataset=hf_dataset)

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        return completion_text

    def _get_example_messages(self, item: dict[str, Any]) -> list[Message]:
        fewshot_examples = self._sample_fewshot_examples(item) if self.num_fewshot > 0 else []

        example_messages = []
        for fewshot_example in fewshot_examples:
            fewshot_example["subject"] = item["subject"]
            example_messages.extend(self._get_instruction_messages(fewshot_example))
            example_messages.append(
                Message(role=Role.ASSISTANT, content=self._get_fewshot_target_text(fewshot_example))
            )
        return example_messages

    def _get_messages(self, item: dict[str, Any]) -> list[Message]:
        example_messages = self._get_example_messages(item)
        instruction_message = self._get_instruction_messages(item)
        cue_text = self._get_cue_text(item)
        cue_message = [Message(role=Role.ASSISTANT, content=cue_text)] if cue_text else []
        messages = example_messages + instruction_message + cue_message
        if initial_prompt_text := self._get_initial_prompt_text(item):
            first_message = messages[0]
            assert first_message.role == Role.USER
            first_message.content = f"{initial_prompt_text}\n\n{first_message.content}"

        if system_prompt_text := self._get_system_prompt_text(item):
            return [Message(role=Role.SYSTEM, content=system_prompt_text)] + messages
        return messages

    def _get_instruction_messages(self, item: dict[str, Any]) -> list[Message]:
        return [Message(role=Role.USER, content=self._get_instruction_text(item))]

    def iterate_samples(self, num_samples: int | None = None) -> Iterable[Sample]:
        for subject in self.SUBJECTS:
            self._load_dataset(subject)
            assert len(self.dataset[self.SAMPLE_SPLIT]) > 0
            done = False
            index = 0
            for item in self.dataset[self.SAMPLE_SPLIT]:
                if done:
                    break
                item["subject"] = subject
                for sample in self._create_samples(item, index, str(subject)):
                    yield sample
                    index += 1
                    if index == num_samples:
                        done = True
                        break

    def _create_samples(self, item: dict[str, Any], index: int, subject: str) -> list[Sample]:
        """Creates one or more samples from a single dataset item. Default implementation returns single sample."""
        return [
            Sample(
                id=index,
                subject=str(subject),
                messages=self._get_messages(item),
                ground_truth=self._get_ground_truth(item),
                possible_completions=self._get_possible_completions(item),
                context=self._get_context(item),
            )
        ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return ""

    def _get_system_prompt_text(self, item: dict[str, Any]) -> str | None:
        return None

    @abstractmethod
    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        raise NotImplementedError

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = self._get_ground_truth(item)
        assert target is not None
        assert isinstance(target, str)
        return target

    @abstractmethod
    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        raise NotImplementedError

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return ""

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return None

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        if self.FEWSHOT_SPLIT == self.SAMPLE_SPLIT:
            fewshot_examples = self.rnd.sample(self.dataset[self.FEWSHOT_SPLIT], self.num_fewshot + 1)
            fewshot_examples = [example for example in fewshot_examples if example != item]
            fewshot_examples = fewshot_examples[: self.num_fewshot]
            return fewshot_examples
        else:
            return self.rnd.sample(self.dataset[self.FEWSHOT_SPLIT], self.num_fewshot)

    def _get_context(self, item: dict[str, Any]) -> BaseMetricContext | list[BaseMetricContext] | None:
        return None

    def get_metadata(self) -> dict[str, str | list[str]]:
        return {
            "dataset_path": self.DATASET_PATH,
            "sample_split": self.SAMPLE_SPLIT,
            "fewshot_split": self.FEWSHOT_SPLIT,
            "response_type": self.RESPONSE_TYPE.value,
            "metrics": [m.NAME for m in self.METRICS],
            "subjects": [str(s) for s in self.SUBJECTS],
        }
