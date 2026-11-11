import logging
import random
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Self, final, override

from eval_framework.answer import AnswerPolicy, PickFromCandidates
from eval_framework.choices import ChoiceReader
from eval_framework.contract import Benchmark, Eval, ResponseType, Sample
from eval_framework.eval_kind import Choice, EvalKind
from eval_framework.fewshot import ChoiceRenderer, FewShot, FewShotGenerator, FewShotPolicy, FewShotSplit, SampleSplit
from eval_framework.shared.types import Completion, RawCompletion
from eval_framework.subjects import NoSubject, Subjects, SubjectsSelector
from eval_framework.tasks.base import RANDOM_SEED
from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy
from eval_framework.tasks.markdown_doc import markdown_doc as render_markdown_doc
from eval_framework.tasks.task_style import Language
from template_formatting.formatter import BaseFormatter, Message, Role

if TYPE_CHECKING:
    from eval_framework.llm.base import BaseLLM
    from eval_framework.metrics.base import BaseMetric
    from eval_framework.tasks.task_style import TaskStyler

logger = logging.getLogger(__name__)

# Deprecated: benchmarks no longer declare a language — ``compose``/``choice`` accept and ignore the
# parameter until all callers stop passing it.
LanguageSpec = Language | dict[str, Language] | None


@final
class ComposedEval(Eval):
    def __init__(
        self,
        *,
        display_name: str,
        kind: EvalKind,
        answer: AnswerPolicy,
        loader: DatasetLoader,
        sample_split: str,
        fewshot: FewShotGenerator,
        subjects: Subjects,
        rnd: random.Random,
    ) -> None:
        self._display_name = display_name
        self._kind = kind
        self._answer = answer
        self.loader = loader
        self.sample_split = sample_split
        self._fewshot = fewshot
        self._subjects = subjects
        self.rnd = rnd

    def _load_dataset(self, load_key: str | None) -> list[dict[str, Any]]:
        """Load and shuffle the sample rows, and let the few-shot generator remember its demonstration pool
        from the same loaded data. Returns the sample rows the eval iterates over."""
        hf_dataset = self.loader.load(load_key)
        sample_rows = list(hf_dataset[self.sample_split])
        self.rnd.shuffle(sample_rows)
        self._fewshot.prepare(hf_dataset, sample_split=self.sample_split, sample_rows=sample_rows)
        return sample_rows

    @override
    def iterate_samples(self, num_samples: int | None = None) -> Iterable[Sample]:
        for subject in self._subjects:
            sample_rows = self._load_dataset(subject.load_key)
            assert len(sample_rows) > 0
            sample_id = 0  # ids and the num_samples cap are per subject
            done = False
            for item in sample_rows:
                if done:
                    break
                item["subject"] = subject.label
                # Draw the few-shot demonstrations once per item, shared across the item's samples.
                fewshot = self._fewshot.for_item(item, self.rnd)
                for sample_body in self._kind.samples(item):
                    yield Sample(
                        id=sample_id,
                        subject=subject.label,
                        messages=self._kind.messages(sample_body, fewshot=fewshot, subject_label=subject.label),
                        ground_truth=sample_body.ground_truth,
                        # An empty candidate list means free-form generation (no candidates to score).
                        possible_completions=sample_body.possible_completions or None,
                        context=sample_body.context,
                    )
                    sample_id += 1
                    if sample_id == num_samples:
                        done = True
                        break

    @override
    def get_metadata(self) -> dict[str, str | list[str]]:
        meta: dict[str, str | list[str]] = {
            "sample_split": self.sample_split,
            "response_type": self.get_response_type().value,
            "metrics": [m.NAME for m in self._kind.metrics()],
            "subjects": [s.label for s in self._subjects],
        }
        meta.update(self._fewshot.metadata(self.sample_split))
        meta.update(self.loader.metadata())
        meta.update(self._kind.metadata())
        return meta

    @override
    def generate_completions(
        self,
        llm: "BaseLLM",
        samples: list[Sample],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> list[Completion]:
        """
        Generates completions for the sample.
        :param sample: sample to generate completions for
        :param stop_sequences: stop sequences to use in completion generation
        :param max_tokens: maximum tokens to use in completion generation
        :return: completion
        """
        if stop_sequences is None:
            stop_sequences = []

        raw_completions: list[RawCompletion]
        raw_completions = llm.generate(samples=samples, stop_sequences=stop_sequences, max_tokens=max_tokens)

        completion_list = []
        for idx, sample in enumerate(samples):
            raw_completion = raw_completions[idx]

            if sample.messages and sample.messages[-1].role == Role.ASSISTANT:
                messages = sample.messages[:-1] + [
                    Message(role=Role.ASSISTANT, content=sample.messages[-1].content + raw_completion.completion)
                ]
            else:
                messages = sample.messages + [Message(role=Role.ASSISTANT, content=raw_completion.completion)]

            # First the model-specific cleanup, then the answer policy's extraction.
            completion = llm.post_process_completion(raw_completion.completion, sample)
            completion = self._answer.extract_answer(
                completion,
                context=sample.context,
                ground_truth=sample.ground_truth,
                messages=sample.messages,
            )

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
                    raw_completion_reasoning=raw_completion.reasoning,
                    raw_completion_finish_reason=raw_completion.finish_reason,
                    context=sample.context,
                    error=raw_completion.raw_completion_error,
                )
            )
        return completion_list

    @override
    def get_stop_sequences(self) -> list[str]:
        return self._answer.stop_sequences()

    @override
    def get_max_tokens(self) -> int | None:
        return self._answer.max_tokens()

    @override
    def get_response_type(self) -> ResponseType:
        return self._answer.response_type()

    @override
    def display_name(self) -> str:
        return self._display_name


@final
class ComposedBenchmark(Benchmark):
    """A ``Benchmark`` that builds a ``ComposedEval`` from an injected eval kind and dataset policy."""

    def __init__(
        self,
        *,
        id: str,
        display_name: str,
        subjects: SubjectsSelector,
        kind: EvalKind,
        answer: AnswerPolicy,
        sample_split: str,
        fewshot: FewShotPolicy,
        dataset_policy: DatasetPolicy,
    ) -> None:
        self._id = id
        self._display_name = display_name
        self._subjects = subjects
        self._kind = kind
        self._answer = answer
        self.sample_split = sample_split
        self._fewshot = fewshot
        self.dataset_policy = dataset_policy

    @classmethod
    def compose(
        cls,
        *,
        id: str,
        kind: EvalKind,
        answer: AnswerPolicy,
        sample_split: str,
        fewshot: FewShotPolicy,
        subjects: SubjectsSelector | None = None,
        dataset_policy: DatasetPolicy,
        language: LanguageSpec = None,
        display_name: str | None = None,
    ) -> Self:
        """Build a ``ComposedBenchmark`` from its inputs; ``subjects`` defaults to ``NoSubject``
        (a single unnamed slice) and ``display_name`` to ``id``. ``language`` is deprecated and ignored."""
        return cls(
            id=id,
            display_name=display_name if display_name is not None else id,
            subjects=subjects if subjects is not None else NoSubject(),
            kind=kind,
            answer=answer,
            sample_split=sample_split,
            fewshot=fewshot,
            dataset_policy=dataset_policy,
        )

    @classmethod
    def choice(
        cls,
        *,
        id: str,
        reader: ChoiceReader,
        styler: "TaskStyler",
        sample_split: str,
        fewshot_split: str,
        subjects: SubjectsSelector | None = None,
        dataset_policy: DatasetPolicy,
        language: LanguageSpec = None,
        display_name: str | None = None,
    ) -> Self:
        """Build a choice-based benchmark. The same ``reader`` + ``styler`` drive both the scored ``Choice``
        and its matching few-shot demonstrations, so they are given once. A choice is always scored by
        loglikelihood over its candidates, so the answer is fixed to ``PickFromCandidates``. The few-shot
        source is picked from whether ``fewshot_split`` is the eval split (leak-safe) or a separate one.
        ``language`` is deprecated and ignored."""
        source = SampleSplit() if fewshot_split == sample_split else FewShotSplit(fewshot_split)
        return cls.compose(
            id=id,
            display_name=display_name,
            kind=Choice(reader, styler),
            answer=PickFromCandidates(),
            sample_split=sample_split,
            fewshot=FewShot(source, ChoiceRenderer(reader, styler)),
            subjects=subjects,
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
        seed: int | None = None,
    ) -> Eval:
        # Bind the shot count into a per-run generator now, before any data loads, so an unsupported request
        # (e.g. few-shot against a 0-shot-only task) fails fast.
        fewshot = self._fewshot.bind(num_fewshot)
        subjects = self._subjects.select(custom_subjects or [])
        if custom_subjects:
            labels = [subject.label for subject in subjects]
            logger.info(f"Restricting subjects to `{labels}` for the task {self._display_name}")
        return ComposedEval(
            display_name=self._display_name,
            kind=self._kind,
            answer=self._answer,
            sample_split=self.sample_split,
            fewshot=fewshot,
            subjects=subjects,
            loader=self.dataset_policy.loader(custom_hf_revision),
            rnd=random.Random(seed),
        )

    @override
    def response_type(self) -> ResponseType:
        """The benchmark's response type"""
        return self._answer.response_type()

    @override
    def metrics(self) -> list[type["BaseMetric"]]:
        """The benchmark's scoring metrics (from the kind) plus the answer's bookkeeping metrics."""
        return self._kind.metrics() + self._answer.metrics()

    @override
    def subjects(self, tokens: list[str] | None = None) -> list[str]:
        """The labels of the subjects the given selector tokens cover (no tokens = all)."""
        return [subject.label for subject in self._subjects.select(tokens or [])]

    @override
    def display_name(self) -> str:
        """The benchmark's human-readable display name."""
        return self._display_name

    @override
    def markdown_doc(self, formatters: Sequence[BaseFormatter]) -> str:
        # The docs show one demonstration where the benchmark samples them, its pinned block where fixed, and
        # none where it is 0-shot only — the policy reports which without a run.
        doc = self._fewshot.documentation(self.sample_split)
        loader = self.dataset_policy.loader(None)
        subjects = self._subjects.select([])
        instance = ComposedEval(
            display_name=self._display_name,
            kind=self._kind,
            answer=self._answer,
            sample_split=self.sample_split,
            fewshot=self._fewshot.bind(doc.example_shots),
            subjects=subjects,
            loader=loader,
            rnd=random.Random(RANDOM_SEED),
        )
        sample = next(iter(instance.iterate_samples(1)))
        loaded = loader.load(subjects[0].load_key)
        split_sizes = {self.sample_split: len(loaded[self.sample_split])}
        if doc.split is not None and doc.split != self.sample_split:
            split_sizes[doc.split] = len(loaded[doc.split])
        return render_markdown_doc(
            name=self._display_name,
            dataset_doc=self.dataset_policy.documentation(),
            sample_split=self.sample_split,
            fewshot_split=doc.split,
            response_type=self.response_type().name,
            metrics=[m.__name__ for m in self.metrics()],
            subjects=[subject.label for subject in subjects],
            num_fewshot=doc.example_shots,
            formatters=formatters,
            example_messages=sample.messages,
            split_sizes=split_sizes,
            possible_completions=sample.possible_completions,
            ground_truth=sample.ground_truth,
        )
