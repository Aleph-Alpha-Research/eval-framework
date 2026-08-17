import logging
import os
import random
import traceback
import typing
from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar

import iso639
from datasets import DatasetDict, DownloadConfig, load_dataset

from eval_framework.metrics.efficiency.bytes_per_sequence_position import (
    BytesCompletion,
    BytesLoglikelihood,
    SequencePositionsCompletion,
    SequencePositionsLoglikelihood,
)
from eval_framework.shared.types import BaseMetricContext, Completion, Error, RawCompletion
from eval_framework.tasks.dataset_revisions import pinned_revision
from eval_framework.tasks.markdown_doc import markdown_doc as render_markdown_doc
from eval_framework.tasks.task import EvalFactory, ResponseType, Sample, Task
from eval_framework.tasks.utils import classproperty, raise_errors
from template_formatting.formatter import BaseFormatter, Message, Role

if TYPE_CHECKING:
    from eval_framework.llm.base import BaseLLM
    from eval_framework.metrics.base import BaseMetric

RANDOM_SEED = 42
NO_SUBJECT = "no_subject"


class TaskStyle(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    CLOZE = "cloze"
    BPB = "bpb"


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


SubjectType = TypeVar("SubjectType")

logger = logging.getLogger(__name__)


class BaseTask[SubjectType](Task):
    NAME: str
    DATASET_PATH: str
    SAMPLE_SPLIT: str
    FEWSHOT_SPLIT: str
    SUBJECTS: list[SubjectType]

    # The lock file this task resolves its pinned dataset revision from, keyed by ``DATASET_PATH``.
    # Each task sets this explicitly: point it at a lock file (e.g. ``HF_REVISIONS_LOCKFILE`` or a
    # frozen one), or ``None`` to opt out of pinning. Deliberately not defaulted so it is never
    # inherited implicitly (a subclass in another package would otherwise resolve the wrong file).
    REVISION_LOCKFILE: Path | None

    # The language (or languages) tested by the benchmark. Accepts a single string, a dictionary specifying
    # language by subtopic, or `None` (for tasks not specific to a single language).
    LANGUAGE: Language | dict[str, Language] | dict[str, tuple[Language, Language]] | None

    # RESPONSE_TYPE and METRICS use exposed as classproperties, so you can access them via either
    # `TaskClass.*` or `task.*` (or `task.get_metrics()`). This avoids mypy conflicts from re-declaring class vars.
    # By default, these values come from TASK_STYLER if set, otherwise from legacy class attributes.

    def __init__(self, num_fewshot: int = 0) -> None:
        self.num_fewshot = num_fewshot
        self.user_prompt_suffix: str | None = None
        self.stop_sequences: list[str] | None = None
        self.max_tokens: int | None = None
        self.hf_revision: str | None = self._apply_hf_revision()
        self.rnd: random.Random | None = None

    def _apply_hf_revision(self, custom_hf_revision: str | None = None) -> str | None:
        # Precedence: CLI/config override > REVISION_LOCKFILE pin.
        # Tasks without a Hugging Face dataset set REVISION_LOCKFILE to None and are not pinned.
        if custom_hf_revision:
            hf_revision = custom_hf_revision
        elif self.REVISION_LOCKFILE is not None:
            hf_revision = pinned_revision(self.REVISION_LOCKFILE, self.DATASET_PATH)
        else:
            hf_revision = None
        return hf_revision

    @classmethod
    def with_overwrite(
        cls,
        num_fewshot: int,
        *,
        custom_subjects: list[str] | None,
        custom_hf_revision: str | None,
        user_prompt_suffix: str | None = None,
        seed: int | None = RANDOM_SEED,
    ) -> Self:
        instance = cls(num_fewshot=num_fewshot)
        if user_prompt_suffix is not None and instance.get_response_type() != ResponseType.COMPLETION:
            raise ValueError("user_prompt_suffix is only supported for completion tasks.")
        instance.user_prompt_suffix = user_prompt_suffix
        instance.rnd = random.Random(seed)
        # If custom subjects were provided during initialization, they take precedence over the class-level SUBJECTS.
        if custom_subjects:
            filtered_subjects = resolve_overwrite_subjects(
                custom_subjects=custom_subjects,
                accepted_subjects=instance.SUBJECTS,
                task_name=instance.display_name(),
            )
            logger.info(f"Setting SUBJECTS to `{filtered_subjects}` for the task {instance.display_name()}")
            instance.SUBJECTS = filtered_subjects

        instance.hf_revision = instance._apply_hf_revision(custom_hf_revision)

        return instance

    def _load_hf_dataset(self, **kwargs: Any) -> Any:
        cache_dir: str = os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets")
        download_config = DownloadConfig(cache_dir=cache_dir, max_retries=5)
        return load_dataset(
            **kwargs,
            revision=self.hf_revision,
            cache_dir=cache_dir,
            download_config=download_config,
        )

    def _shuffle_splits(self, hf_dataset: DatasetDict) -> dict[str, Any]:
        assert self.rnd is not None, "Task RNG is unseeded; build tasks via `with_overwrite`."
        dataset = {}

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
        instruction_message = self._apply_user_prompt_suffix(self._get_instruction_messages(item))
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

    def _apply_user_prompt_suffix(self, instruction_messages: list[Message]) -> list[Message]:
        """Append the configured suffix verbatim to the evaluated user turn."""
        if self.user_prompt_suffix is None:
            return instruction_messages

        for message in reversed(instruction_messages):
            if message.role == Role.USER:
                message.content = f"{message.content}{self.user_prompt_suffix}"
                return instruction_messages

        raise ValueError("Cannot append user_prompt_suffix: evaluated instruction contains no user message.")

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

    def markdown_doc(self, formatters: Sequence[BaseFormatter]) -> str:
        """Render this task's documentation as markdown."""
        dataset_path = getattr(self, "DATASET_PATH", None)
        example_messages = split_sizes = possible_completions = ground_truth = None
        if dataset_path is None:
            sample = next(iter(self.iterate_samples(1)))
            example_messages = sample.messages
            split_sizes = {split: len(self.dataset[split]) for split in self.dataset}
            possible_completions = sample.possible_completions
            ground_truth = sample.ground_truth

        return render_markdown_doc(
            name=self.NAME,
            dataset_path=dataset_path,
            sample_split=getattr(self, "SAMPLE_SPLIT", None),
            fewshot_split=getattr(self, "FEWSHOT_SPLIT", None),
            response_type=self.get_response_type().name,
            metrics=[m.__name__ for m in self.get_metrics()],
            subjects=getattr(self, "SUBJECTS", None),
            language=getattr(self, "LANGUAGE", None),
            num_fewshot=self.num_fewshot,
            formatters=formatters,
            example_messages=example_messages,
            split_sizes=split_sizes,
            possible_completions=possible_completions,
            ground_truth=ground_truth,
        )

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

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        raise NotImplementedError("Subclasses using a TASK_STYLER must implement _get_raw_question")

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        raise NotImplementedError("Subclasses using a TASK_STYLER must implement _get_choices")

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        raise NotImplementedError("Subclasses using a TASK_STYLER must implement _get_correct_index")

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        if hasattr(self, "TASK_STYLER"):
            return self.TASK_STYLER.get_instruction_text(self._get_raw_question(item), self._get_choices(item))
        raise NotImplementedError

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        if hasattr(self, "TASK_STYLER"):
            return self.TASK_STYLER.get_fewshot_target_text(self._get_choices(item), self._get_correct_index(item))
        target = self._get_ground_truth(item)
        assert target is not None
        assert isinstance(target, str)
        return target

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        if hasattr(self, "TASK_STYLER"):
            return self.TASK_STYLER.get_ground_truth(self._get_choices(item), self._get_correct_index(item))
        raise NotImplementedError

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        if hasattr(self, "TASK_STYLER"):
            return self.TASK_STYLER.get_cue_text()
        return ""

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        if hasattr(self, "TASK_STYLER"):
            return self.TASK_STYLER.get_possible_completions(self._get_choices(item), self._get_correct_index(item))
        return None

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        assert self.rnd is not None, "Task RNG is unseeded; build tasks via `with_overwrite`."
        if self.FEWSHOT_SPLIT == self.SAMPLE_SPLIT:
            # If the fewshot and sample splits are the same, we risk including the current eval item
            # as a fewshot example (leaking the answer). To prevent this, sample one extra example,
            # remove the current item if present, and truncate back to num_fewshot.
            fewshot_examples = self.rnd.sample(self.dataset[self.FEWSHOT_SPLIT], self.num_fewshot + 1)
            fewshot_examples = [example for example in fewshot_examples if example != item]
            fewshot_examples = fewshot_examples[: self.num_fewshot]
            return fewshot_examples
        else:
            # Separate splits: no risk of leaking the current item, sample directly.
            return self.rnd.sample(self.dataset[self.FEWSHOT_SPLIT], self.num_fewshot)

    def _get_context(self, item: dict[str, Any]) -> BaseMetricContext | list[BaseMetricContext] | None:
        return None

    def get_metadata(self) -> dict[str, str | list[str]]:
        meta: dict[str, str | list[str]] = {
            "dataset_path": self.DATASET_PATH,
            "sample_split": self.SAMPLE_SPLIT,
            "fewshot_split": self.FEWSHOT_SPLIT,
            "response_type": self.get_response_type().value,
            "metrics": [m.NAME for m in self.get_metrics()],
            "subjects": [str(s) for s in self.SUBJECTS],
        }
        if hasattr(self, "TASK_STYLER"):
            meta.update(self.TASK_STYLER.get_extra_metadata())
        return meta

    def generate_completions(
        self,
        llm: "BaseLLM",
        samples: list[Sample],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        fail_on_error: bool = True,
    ) -> list[Completion]:
        """
        Generates completions for the sample.
        :param sample: sample to generate completions for
        :param stop_sequences: stop sequences to use in completion generation
        :param max_tokens: maximum tokens to use in completion generation
        :param fail_on_error: if True, re-raise the original exception instead of capturing it
                              into a per-sample Error completion
        :return: completion
        """
        if stop_sequences is None:
            stop_sequences = []

        raw_completions: list[RawCompletion]
        try:
            raw_completions = llm.generate(samples=samples, stop_sequences=stop_sequences, max_tokens=max_tokens)
        except Exception as e:
            if raise_errors() or fail_on_error:
                raise
            logger.info(f"Error: {e.__class__.__name__} {e}")
            raw_completions = [
                RawCompletion(
                    prompt="",
                    prompt_sequence_positions=0,
                    completion="",
                    completion_sequence_positions=0,
                    raw_completion_error=Error(
                        error_class=e.__class__.__name__, message=str(e), traceback=traceback.format_exc()
                    ),
                )
                for _ in range(len(samples))
            ]

        completion_list = []
        for idx, sample in enumerate(samples):
            raw_completion = raw_completions[idx]

            if sample.messages and sample.messages[-1].role == Role.ASSISTANT:
                messages = sample.messages[:-1] + [
                    Message(role=Role.ASSISTANT, content=sample.messages[-1].content + raw_completion.completion)
                ]
            else:
                messages = sample.messages + [Message(role=Role.ASSISTANT, content=raw_completion.completion)]

            try:
                error = None
                model_post_processed_completion = llm.post_process_completion(raw_completion.completion, sample)
                completion = self.post_process_generated_completion(model_post_processed_completion, sample)
            except Exception as e:
                if raise_errors() or fail_on_error:
                    raise
                error = Error(error_class=e.__class__.__name__, message=str(e), traceback=traceback.format_exc())
                completion = ""

            completion_list.append(
                Completion(
                    id=sample.id,
                    subject=sample.subject,
                    ground_truth=sample.ground_truth,
                    prompt=raw_completion.prompt,
                    prompt_sequence_positions=raw_completion.prompt_sequence_positions,
                    concat_compression=raw_completion.concat_compression,
                    messages=messages,
                    completion=completion,
                    raw_completion=raw_completion.completion,
                    raw_completion_sequence_positions=raw_completion.completion_sequence_positions,
                    context=sample.context,
                    error=raw_completion.raw_completion_error or error,
                )
            )
        return completion_list

    @classmethod
    def get_response_type(cls) -> ResponseType:
        """Return the response type of the task (or the styler if it exists)."""
        if hasattr(cls, "TASK_STYLER"):
            return cls.TASK_STYLER.response_type
        return cls.RESPONSE_TYPE

    @classmethod
    def get_metrics(cls) -> list[type["BaseMetric"]]:
        """Return the metrics of the task (or the styler if it exists)."""
        if hasattr(cls, "TASK_STYLER"):
            task_metrics = cls.TASK_STYLER.metrics
        else:
            task_metrics = cls.METRICS

        match cls.get_response_type():
            case ResponseType.COMPLETION:
                metrics = task_metrics + [BytesCompletion, SequencePositionsCompletion]
            case ResponseType.LOGLIKELIHOODS:
                metrics = task_metrics + [BytesLoglikelihood, SequencePositionsLoglikelihood]
            case _:
                typing.assert_never(cls.get_response_type())

        return metrics

    @classproperty
    def RESPONSE_TYPE(cls) -> ResponseType:
        """For backwards compatibility."""
        return cls.get_response_type()

    @classproperty
    def METRICS(cls) -> list[type["BaseMetric"]]:
        """For backwards compatibility."""
        return cls.get_metrics()

    def display_name(self) -> str:
        return self.NAME


def _subject_parts(subject: object) -> tuple[str, ...]:
    """A subject as its stringified parts: tuple subjects keep their fields, scalars are a single part.

    Comparing string forms means tuple fields of any type work (e.g. tuple[str, int, str]) without
    having to parse the CLI token into each field's native type.
    """
    return tuple(str(field) for field in subject) if isinstance(subject, tuple) else (str(subject),)


def resolve_overwrite_subjects[SubjectType](
    custom_subjects: list[str], accepted_subjects: list[SubjectType], task_name: str
) -> list[SubjectType]:
    """Restrict `accepted_subjects` to the ones requested via --task-subjects.

    A subject matches a token when their parts agree position by position, where "*" matches any one
    part -- so "DE_DE,*" selects every German subject and "*" selects every scalar subject.
    """
    if not accepted_subjects:
        raise ValueError(f"Task {task_name} has no SUBJECTS defined")

    filters = {token: tuple(part.strip() for part in token.split(",")) for token in custom_subjects}

    # Select and validate in one pass. Every filter is tried against every subject (no early exit), so
    # a filter still counts as used when its subjects were already selected by an earlier filter --
    # e.g. "a,*" after "a,1" is valid, not unused.
    chosen_subjects: list[SubjectType] = []
    used_filters: set[str] = set()
    for subject in accepted_subjects:
        fields = _subject_parts(subject)
        matching = [
            token
            for token, parts in filters.items()
            if len(parts) == len(fields) and all(p in ("*", f) for p, f in zip(parts, fields))
        ]
        if matching:
            chosen_subjects.append(subject)
            used_filters.update(matching)

    for token in custom_subjects:
        if token not in used_filters:
            raise ValueError(
                f"Subject '{token}' not found in task {task_name}. Subjects are matched by their "
                f"string form, so check number and enum formatting."
            )

    return chosen_subjects


class Eager(EvalFactory):
    """Wraps an already-imported task class."""

    def __init__(self, task: type[BaseTask]) -> None:
        self._task = task

    def id(self) -> str:
        return self._task.__name__

    def create(
        self,
        num_fewshot: int,
        custom_subjects: list[str] | None,
        custom_hf_revision: str | None,
        user_prompt_suffix: str | None = None,
        seed: int | None = None,
    ) -> BaseTask:
        return self._task.with_overwrite(
            num_fewshot=num_fewshot,
            custom_subjects=custom_subjects,
            custom_hf_revision=custom_hf_revision,
            user_prompt_suffix=user_prompt_suffix,
            seed=seed,
        )

    def response_type(self) -> ResponseType:
        """The eval's response type"""
        return self._task.get_response_type()

    def metrics(self) -> list[type["BaseMetric"]]:
        """The eval's metrics"""
        return self._task.get_metrics()

    def subjects(self) -> list[Any]:
        """The eval's subjects"""
        return self._task.SUBJECTS

    def display_name(self) -> str:
        """The eval's human-readable display name (the task's ``NAME``)."""
        return self._task.NAME

    def markdown_doc(self, formatters: Sequence[BaseFormatter]) -> str:
        try:
            task = self.create(num_fewshot=1, custom_subjects=None, custom_hf_revision=None, seed=RANDOM_SEED)
        except (TypeError, ValueError, AssertionError):
            task = self.create(num_fewshot=0, custom_subjects=None, custom_hf_revision=None, seed=RANDOM_SEED)
        return task.markdown_doc(formatters)
