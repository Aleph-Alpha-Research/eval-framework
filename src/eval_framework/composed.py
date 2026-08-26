import logging
import random
import traceback
import typing
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, final, override

from datasets import DatasetDict

from eval_framework.contract import Benchmark, Eval, ResponseType, Sample
from eval_framework.metrics.efficiency.bytes_per_sequence_position import (
    BytesCompletion,
    BytesLoglikelihood,
    SequencePositionsCompletion,
    SequencePositionsLoglikelihood,
)
from eval_framework.metrics.efficiency.token_counters import TokenCounts
from eval_framework.shared.errors import raise_errors
from eval_framework.shared.types import Completion, Error, RawCompletion
from eval_framework.subjects import Subjects, SubjectsSelector
from eval_framework.tasks.base import RANDOM_SEED, Language
from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy
from eval_framework.tasks.markdown_doc import markdown_doc as render_markdown_doc
from template_formatting.formatter import BaseFormatter, Message, Role

if TYPE_CHECKING:
    from eval_framework.llm.base import BaseLLM
    from eval_framework.metrics.base import BaseMetric
    from eval_framework.tasks.task_style import TaskStyler

logger = logging.getLogger(__name__)

# The language(s) a benchmark tests: a single language, a per-subtopic mapping, or None (not language-specific).
LanguageSpec = Language | dict[str, Language] | dict[str, tuple[Language, Language]] | None


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


@final
class ComposedEval(Eval):
    def __init__(
        self,
        num_fewshot: int = 0,
        *,
        display_name: str,
        reader: ChoiceReader,
        loader: DatasetLoader,
        styler: "TaskStyler",
        sample_split: str,
        fewshot_split: str,
        subjects: Subjects,
        language: LanguageSpec,
        rnd: random.Random,
    ) -> None:
        self._display_name = display_name
        self.num_fewshot = num_fewshot
        self.reader = reader
        self.loader = loader
        self.styler = styler
        self.sample_split = sample_split
        self.fewshot_split = fewshot_split
        self._subjects = subjects
        self.language = language
        self.rnd = rnd

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

    def _load_dataset(self, load_key: str | None) -> dict[str, Any]:
        hf_dataset = self.loader.load(load_key)
        return self._shuffle_splits(hf_dataset=hf_dataset)

    def _example_messages(self, item: dict[str, Any], fewshot_pool: list[dict]) -> list[Message]:
        fewshot_examples = self._sample_fewshot_examples(item, fewshot_pool) if self.num_fewshot > 0 else []

        example_messages = []
        for fewshot_example in fewshot_examples:
            fewshot_example["subject"] = item["subject"]
            example_messages.extend(self._get_instruction_messages(fewshot_example))
            example_messages.append(
                Message(role=Role.ASSISTANT, content=self._get_fewshot_target_text(fewshot_example))
            )
        return example_messages

    def _messages(self, item: dict[str, Any], fewshot_pool: list[dict]) -> list[Message]:
        example_messages = self._example_messages(item, fewshot_pool)
        instruction_message = self._get_instruction_messages(item)
        cue_text = self._get_cue_text(item)
        cue_message = [Message(role=Role.ASSISTANT, content=cue_text)] if cue_text else []
        return example_messages + instruction_message + cue_message

    def _get_instruction_messages(self, item: dict[str, Any]) -> list[Message]:
        return [Message(role=Role.USER, content=self._get_instruction_text(item))]

    @override
    def iterate_samples(self, num_samples: int | None = None) -> Iterable[Sample]:
        for subject in self._subjects:
            dataset = self._load_dataset(subject.load_key)
            fewshot_pool = dataset[self.fewshot_split] if self.num_fewshot > 0 else []
            assert len(dataset[self.sample_split]) > 0
            for index, item in enumerate(dataset[self.sample_split]):
                if index == num_samples:
                    break
                item["subject"] = subject.label
                yield self._create_sample(item, index, subject.label, fewshot_pool)

    def _create_sample(self, item: dict[str, Any], index: int, subject: str, fewshot_pool: list[dict]) -> Sample:
        return Sample(
            id=index,
            subject=str(subject),
            messages=self._messages(item, fewshot_pool),
            ground_truth=self._get_ground_truth(item),
            possible_completions=self._get_possible_completions(item),
            context=None,
        )

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        fields = self.reader.read(item)
        return self.styler.get_instruction_text(fields.raw_question, fields.choices)

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        fields = self.reader.read(item)
        return self.styler.get_fewshot_target_text(fields.choices, fields.correct_index)

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        fields = self.reader.read(item)
        return self.styler.get_ground_truth(fields.choices, fields.correct_index)

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return self.styler.get_cue_text()

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        fields = self.reader.read(item)
        return self.styler.get_possible_completions(fields.choices, fields.correct_index)

    def _sample_fewshot_examples(self, item: dict[str, Any], fewshot_pool: list[dict]) -> list[dict]:
        if self.fewshot_split == self.sample_split:
            # If the fewshot and sample splits are the same, we risk including the current eval item
            # as a fewshot example (leaking the answer). To prevent this, sample one extra example,
            # remove the current item if present, and truncate back to num_fewshot.
            fewshot_examples = self.rnd.sample(fewshot_pool, self.num_fewshot + 1)
            fewshot_examples = [example for example in fewshot_examples if example != item]
            fewshot_examples = fewshot_examples[: self.num_fewshot]
            return fewshot_examples
        else:
            # Separate splits: no risk of leaking the current item, sample directly.
            return self.rnd.sample(fewshot_pool, self.num_fewshot)

    @override
    def get_metadata(self) -> dict[str, str | list[str]]:
        meta: dict[str, str | list[str]] = {
            "sample_split": self.sample_split,
            "fewshot_split": self.fewshot_split,
            "response_type": self.get_response_type().value,
            "metrics": [m.NAME for m in self.styler.metrics],
            "subjects": [s.label for s in self._subjects],
        }
        meta.update(self.loader.metadata())
        meta.update(self.styler.get_extra_metadata())
        return meta

    @override
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
                completion = llm.post_process_completion(raw_completion.completion, sample)
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

    @override
    def get_response_type(self) -> ResponseType:
        return self.styler.response_type

    @override
    def display_name(self) -> str:
        return self._display_name


def _metrics_for(styler: "TaskStyler") -> list[type["BaseMetric"]]:
    """The metrics a styler implies: its own plus those its response type requires."""
    response_type_metrics: list[type[BaseMetric]]
    match styler.response_type:
        case ResponseType.COMPLETION:
            response_type_metrics = [BytesCompletion, SequencePositionsCompletion, TokenCounts]
        case ResponseType.LOGLIKELIHOODS:
            response_type_metrics = [BytesLoglikelihood, SequencePositionsLoglikelihood]
        case _:
            typing.assert_never(styler.response_type)
    return styler.metrics + response_type_metrics


@final
class ComposedBenchmark(Benchmark):
    """A ``Benchmark`` that builds a ``ComposedEval`` from an injected styler and dataset inputs."""

    def __init__(
        self,
        *,
        id: str,
        display_name: str,
        subjects: SubjectsSelector,
        styler: "TaskStyler",
        reader: ChoiceReader,
        sample_split: str,
        fewshot_split: str,
        dataset_policy: DatasetPolicy,
        language: LanguageSpec,
    ) -> None:
        self._id = id
        self._display_name = display_name
        self._subjects = subjects
        self._styler = styler
        self.reader = reader
        self.sample_split = sample_split
        self.fewshot_split = fewshot_split
        self.language = language
        self.dataset_policy = dataset_policy

    @classmethod
    def compose(
        cls,
        *,
        id: str,
        styler: "TaskStyler",
        reader: ChoiceReader,
        sample_split: str,
        fewshot_split: str,
        subjects: SubjectsSelector,
        dataset_policy: DatasetPolicy,
        language: LanguageSpec,
        display_name: str | None = None,
    ) -> Self:
        """Build a ``ComposedBenchmark`` from its inputs; ``display_name`` defaults to ``id``."""
        return cls(
            id=id,
            display_name=display_name if display_name is not None else id,
            subjects=subjects,
            styler=styler,
            reader=reader,
            sample_split=sample_split,
            fewshot_split=fewshot_split,
            language=language,
            dataset_policy=dataset_policy,
        )

    @override
    def id(self) -> str:
        return self._id

    @override
    def create(
        self,
        num_fewshot: int,
        custom_subjects: list[str] | None,
        custom_hf_revision: str | None,
        user_prompt_suffix: str | None = None,
        seed: int | None = None,
    ) -> Eval:
        # Composed evals have no completion path yet, so a completion-only user prompt suffix is rejected.
        if user_prompt_suffix is not None:
            raise ValueError("user_prompt_suffix is only supported for completion tasks.")
        subjects = self._subjects.select(custom_subjects or [])
        if custom_subjects:
            labels = [subject.label for subject in subjects]
            logger.info(f"Restricting subjects to `{labels}` for the task {self._display_name}")
        return ComposedEval(
            num_fewshot=num_fewshot,
            display_name=self._display_name,
            reader=self.reader,
            styler=self._styler,
            sample_split=self.sample_split,
            fewshot_split=self.fewshot_split,
            subjects=subjects,
            language=self.language,
            loader=self.dataset_policy.loader(custom_hf_revision),
            rnd=random.Random(seed),
        )

    @override
    def response_type(self) -> ResponseType:
        """The benchmark's response type"""
        return self._styler.response_type

    @override
    def metrics(self) -> list[type["BaseMetric"]]:
        """The benchmark's metrics"""
        return _metrics_for(self._styler)

    @override
    def subjects(self) -> list[Any]:
        """The benchmark's subjects"""
        return [subject.label for subject in self._subjects.select([])]

    @override
    def display_name(self) -> str:
        """The benchmark's human-readable display name."""
        return self._display_name

    @override
    def markdown_doc(self, formatters: Sequence[BaseFormatter]) -> str:
        num_fewshot = 1
        subjects = self._subjects.select([])
        instance = ComposedEval(
            num_fewshot=num_fewshot,
            display_name=self._display_name,
            reader=self.reader,
            styler=self._styler,
            sample_split=self.sample_split,
            fewshot_split=self.fewshot_split,
            subjects=subjects,
            language=self.language,
            loader=self.dataset_policy.loader(None),
            rnd=random.Random(RANDOM_SEED),
        )
        sample = next(iter(instance.iterate_samples(1)))
        dataset = instance._load_dataset(subjects[0].load_key)
        return render_markdown_doc(
            name=self._display_name,
            dataset_doc=self.dataset_policy.documentation(),
            sample_split=self.sample_split,
            fewshot_split=self.fewshot_split,
            response_type=self.response_type().name,
            metrics=[m.__name__ for m in self.metrics()],
            subjects=[subject.label for subject in subjects],
            language=self.language,
            num_fewshot=num_fewshot,
            formatters=formatters,
            example_messages=sample.messages,
            split_sizes={split: len(dataset[split]) for split in dataset},
            possible_completions=sample.possible_completions,
            ground_truth=sample.ground_truth,
        )
