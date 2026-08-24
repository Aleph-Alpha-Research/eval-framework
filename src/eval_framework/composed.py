import logging
import os
import random
import traceback
import typing
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from datasets import DatasetDict, DownloadConfig, load_dataset

from eval_framework.contract import Benchmark, Eval, ResponseType, Sample
from eval_framework.metrics.efficiency.bytes_per_sequence_position import (
    BytesCompletion,
    BytesLoglikelihood,
    SequencePositionsCompletion,
    SequencePositionsLoglikelihood,
)
from eval_framework.metrics.efficiency.token_counters import TokenCounts
from eval_framework.shared.errors import raise_errors
from eval_framework.shared.types import BaseMetricContext, Completion, Error, RawCompletion
from eval_framework.tasks.base import (
    NO_SUBJECT,
    RANDOM_SEED,
    Language,
    resolve_overwrite_subjects,
)
from eval_framework.tasks.dataset_revisions import pinned_revision
from eval_framework.tasks.markdown_doc import markdown_doc as render_markdown_doc
from template_formatting.formatter import BaseFormatter, Message, Role

if TYPE_CHECKING:
    from eval_framework.llm.base import BaseLLM
    from eval_framework.metrics.base import BaseMetric
    from eval_framework.tasks.task_style import TaskStyler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChoiceFields:
    """The fields a choice-based styler needs out of a single dataset item.

    ``choices`` and ``correct_index`` are produced together (a benchmark may shuffle the correct
    answer in among distractors), so a reader yields them in one ``read`` rather than via separate
    calls that would each have to re-derive the same shuffle.
    """

    raw_question: str
    choices: list[str]
    correct_index: int


class ChoiceReader(ABC):
    """Reads the fields a styler needs out of a raw dataset item.

    Isolates dataset-schema knowledge here, so neither the eval nor the styler has to know the shape
    of a benchmark's items.
    """

    @abstractmethod
    def read(self, item: dict[str, Any]) -> ChoiceFields: ...


class ComposedEval[SubjectType](Eval):
    NAME: str

    # Composed evals are styler-only (choice-based); there is no completion styling path.
    TASK_STYLER: "TaskStyler"

    # The lock file this task resolves its pinned dataset revision from, keyed by ``dataset_path``.
    # Each task sets this explicitly: point it at a lock file (e.g. ``HF_REVISIONS_LOCKFILE`` or a
    # frozen one), or ``None`` to opt out of pinning. Deliberately not defaulted so it is never
    # inherited implicitly (a subclass in another package would otherwise resolve the wrong file).
    REVISION_LOCKFILE: Path | None

    # The language (or languages) tested by the benchmark. Accepts a single string, a dictionary specifying
    # language by subtopic, or `None` (for tasks not specific to a single language).
    LANGUAGE: Language | dict[str, Language] | dict[str, tuple[Language, Language]] | None

    def __init__(
        self,
        num_fewshot: int = 0,
        *,
        reader: ChoiceReader,
        dataset_path: str,
        sample_split: str,
        fewshot_split: str,
        subjects: list[SubjectType],
        custom_hf_revision: str | None = None,
        user_prompt_suffix: str | None = None,
        seed: int | None = RANDOM_SEED,
    ) -> None:
        # No completion path yet, so any suffix is rejected. This deliberately trips even for a
        # completion response type, flagging that suffix handling must be implemented once that lands.
        if user_prompt_suffix is not None:
            raise ValueError("user_prompt_suffix is only supported for completion tasks.")

        self.num_fewshot = num_fewshot
        self.reader = reader
        self.dataset_path = dataset_path
        self.sample_split = sample_split
        self.fewshot_split = fewshot_split
        self.subjects = subjects
        self.rnd = random.Random(seed)
        self.hf_revision: str | None = self._apply_hf_revision(custom_hf_revision)

    def _apply_hf_revision(self, custom_hf_revision: str | None = None) -> str | None:
        # Precedence: CLI/config override > REVISION_LOCKFILE pin.
        # Tasks without a Hugging Face dataset set REVISION_LOCKFILE to None and are not pinned.
        if custom_hf_revision:
            hf_revision = custom_hf_revision
        elif self.REVISION_LOCKFILE is not None:
            hf_revision = pinned_revision(self.REVISION_LOCKFILE, self.dataset_path)
        else:
            hf_revision = None
        return hf_revision

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
        dataset = {}

        for split, data in hf_dataset.items():
            if split not in [self.sample_split, self.fewshot_split]:
                continue

            data_list = list(data)

            if split == self.sample_split:
                self.rnd.shuffle(data_list)

            dataset[split] = data_list

        return dataset

    def _load_dataset(self, subject: SubjectType) -> None:
        name = subject if subject != NO_SUBJECT else None
        hf_dataset = self._load_hf_dataset(path=self.dataset_path, name=name)
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
        for subject in self.subjects:
            self._load_dataset(subject)
            assert len(self.dataset[self.sample_split]) > 0
            done = False
            index = 0
            for item in self.dataset[self.sample_split]:
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
        dataset_path = self.dataset_path
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
            sample_split=self.sample_split,
            fewshot_split=self.fewshot_split,
            response_type=self.get_response_type().name,
            metrics=[m.__name__ for m in self.get_metrics()],
            subjects=self.subjects,
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

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        fields = self.reader.read(item)
        return self.TASK_STYLER.get_instruction_text(fields.raw_question, fields.choices)

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        fields = self.reader.read(item)
        return self.TASK_STYLER.get_fewshot_target_text(fields.choices, fields.correct_index)

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        fields = self.reader.read(item)
        return self.TASK_STYLER.get_ground_truth(fields.choices, fields.correct_index)

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return self.TASK_STYLER.get_cue_text()

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        fields = self.reader.read(item)
        return self.TASK_STYLER.get_possible_completions(fields.choices, fields.correct_index)

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        if self.fewshot_split == self.sample_split:
            # If the fewshot and sample splits are the same, we risk including the current eval item
            # as a fewshot example (leaking the answer). To prevent this, sample one extra example,
            # remove the current item if present, and truncate back to num_fewshot.
            fewshot_examples = self.rnd.sample(self.dataset[self.fewshot_split], self.num_fewshot + 1)
            fewshot_examples = [example for example in fewshot_examples if example != item]
            fewshot_examples = fewshot_examples[: self.num_fewshot]
            return fewshot_examples
        else:
            # Separate splits: no risk of leaking the current item, sample directly.
            return self.rnd.sample(self.dataset[self.fewshot_split], self.num_fewshot)

    def _get_context(self, item: dict[str, Any]) -> BaseMetricContext | list[BaseMetricContext] | None:
        return None

    def get_metadata(self) -> dict[str, str | list[str]]:
        meta: dict[str, str | list[str]] = {
            "dataset_path": self.dataset_path,
            "sample_split": self.sample_split,
            "fewshot_split": self.fewshot_split,
            "response_type": self.get_response_type().value,
            "metrics": [m.NAME for m in self._get_task_specific_metrics()],
            "subjects": [str(s) for s in self.subjects],
        }
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
                    prompt_num_tokens=0,
                    completion="",
                    completion_num_tokens=0,
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
                    prompt_num_tokens=raw_completion.prompt_num_tokens,
                    concat_compression=raw_completion.concat_compression,
                    messages=messages,
                    completion=completion,
                    raw_completion=raw_completion.completion,
                    raw_completion_num_tokens=raw_completion.completion_num_tokens,
                    raw_completion_reasoning_num_tokens=raw_completion.reasoning_num_tokens,
                    context=sample.context,
                    error=raw_completion.raw_completion_error or error,
                )
            )
        return completion_list

    @classmethod
    def get_response_type(cls) -> ResponseType:
        return cls.TASK_STYLER.response_type

    @classmethod
    def _get_task_specific_metrics(cls) -> list[type["BaseMetric"]]:
        return cls.TASK_STYLER.metrics

    @classmethod
    def _get_response_type_specific_metrics(cls) -> list[type["BaseMetric"]]:
        metrics: list[type[BaseMetric]]
        match cls.get_response_type():
            case ResponseType.COMPLETION:
                metrics = [
                    BytesCompletion,
                    SequencePositionsCompletion,
                    TokenCounts,
                ]
            case ResponseType.LOGLIKELIHOODS:
                metrics = [BytesLoglikelihood, SequencePositionsLoglikelihood]
            case _:
                typing.assert_never(cls.get_response_type())

        return metrics

    @classmethod
    def get_metrics(cls) -> list[type["BaseMetric"]]:
        task_metrics = cls._get_task_specific_metrics()
        response_type_metrics = cls._get_response_type_specific_metrics()

        return task_metrics + response_type_metrics

    def display_name(self) -> str:
        return self.NAME


class ComposedBenchmark(Benchmark):
    """A ``Benchmark`` that constructs one ``ComposedEval`` type from precomputed metadata + inputs."""

    def __init__(
        self,
        *,
        id: str,
        display_name: str,
        subjects: list[Any],
        metrics: list[type["BaseMetric"]],
        response_type: ResponseType,
        reader: ChoiceReader,
        dataset_path: str,
        sample_split: str,
        fewshot_split: str,
        task: type[ComposedEval],
    ) -> None:
        self._id = id
        self._display_name = display_name
        self._subjects = subjects
        self._metrics = metrics
        self._response_type = response_type
        self.reader = reader
        self.dataset_path = dataset_path
        self.sample_split = sample_split
        self.fewshot_split = fewshot_split
        self._task = task

    @classmethod
    def from_base(
        cls,
        task: type[ComposedEval],
        reader: ChoiceReader,
        dataset_path: str,
        sample_split: str,
        fewshot_split: str,
        subjects: list[Any],
    ) -> Self:
        """Build an ``ComposedBenchmark`` from a task class and its construction inputs.

        Metadata (name, metrics, response type) is derived from the class.
        """
        return cls(
            id=task.__name__,
            display_name=task.NAME,
            subjects=subjects,
            metrics=task.get_metrics(),
            response_type=task.get_response_type(),
            reader=reader,
            dataset_path=dataset_path,
            sample_split=sample_split,
            fewshot_split=fewshot_split,
            task=task,
        )

    def id(self) -> str:
        return self._id

    def create(
        self,
        num_fewshot: int,
        custom_subjects: list[str] | None,
        custom_hf_revision: str | None,
        user_prompt_suffix: str | None = None,
        seed: int | None = None,
    ) -> Eval:
        subjects = self._subjects
        if custom_subjects:
            subjects = resolve_overwrite_subjects(
                custom_subjects=custom_subjects, accepted_subjects=self._subjects, task_name=self._display_name
            )
            logger.info(f"Restricting subjects to `{subjects}` for the task {self._display_name}")
        return self._task(
            num_fewshot=num_fewshot,
            reader=self.reader,
            dataset_path=self.dataset_path,
            sample_split=self.sample_split,
            fewshot_split=self.fewshot_split,
            subjects=subjects,
            custom_hf_revision=custom_hf_revision,
            user_prompt_suffix=user_prompt_suffix,
            seed=seed,
        )

    def response_type(self) -> ResponseType:
        """The eval's response type"""
        return self._response_type

    def metrics(self) -> list[type["BaseMetric"]]:
        """The eval's metrics"""
        return self._metrics

    def subjects(self) -> list[Any]:
        """The eval's subjects"""
        return self._subjects

    def display_name(self) -> str:
        """The eval's human-readable display name."""
        return self._display_name

    def markdown_doc(self, formatters: Sequence[BaseFormatter]) -> str:
        instance = self._task(
            num_fewshot=1,
            reader=self.reader,
            dataset_path=self.dataset_path,
            sample_split=self.sample_split,
            fewshot_split=self.fewshot_split,
            subjects=self._subjects,
            custom_hf_revision=None,
            seed=RANDOM_SEED,
        )
        return instance.markdown_doc(formatters)
