import random
from typing import Any, override
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark, ComposedEval, LanguageSpec
from eval_framework.contract import ResponseType
from eval_framework.eval_kind import Choice
from eval_framework.metrics.base import BaseMetric
from eval_framework.metrics.efficiency.bytes_per_sequence_position import (
    BytesLoglikelihood,
    SequencePositionsLoglikelihood,
)
from eval_framework.run import parse_args
from eval_framework.subjects import ListOfSubjects, Subject, Subjects, SubjectsSelector
from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy
from eval_framework.tasks.task_style import TaskStyle, TaskStyler
from template_formatting.formatter import ConcatFormatter, Message, Role


class _DummyReader(ChoiceReader):
    """A dummy reader: passed only to satisfy construction for doubles that never read (they override
    the styling methods, or raise before styling)."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(raw_question="", choices=[], correct_index=0)


class _DummyDatasetLoader(DatasetLoader):
    """A no-op loader for doubles that never load a dataset."""

    @override
    def load(self, name: str | None) -> DatasetDict:
        return DatasetDict()

    @override
    def metadata(self) -> dict[str, str]:
        return {}


class _DummyDatasetPolicy(DatasetPolicy):
    """A no-op policy for benchmark doubles that never load a dataset."""

    @override
    def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
        return _DUMMY_LOADER

    @override
    def documentation(self) -> str:
        return ""


_DUMMY_READER = _DummyReader()
_DUMMY_LOADER = _DummyDatasetLoader()
_DUMMY_RNG = random.Random(0)
_DUMMY_SPLIT = "test"
_DUMMY_SELECTOR: SubjectsSelector = ListOfSubjects(["subject"])
_DUMMY_EVAL_SUBJECTS: Subjects = (Subject(load_key="subject", label="subject"),)


class _DummyStyler(TaskStyler):
    """A dummy styler for tests that need a ComposedEval with a (loglikelihood) styler but never render
    a prompt — e.g. asserting a user_prompt_suffix is rejected."""

    response_type = ResponseType.LOGLIKELIHOODS
    metrics: list[type[BaseMetric]] = []
    task_style = TaskStyle.MULTIPLE_CHOICE
    question_prefix = ""

    @override
    def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
        return ""

    @override
    def get_ground_truth(self, choices: list[str], correct_index: int) -> str:
        return ""

    @override
    def get_possible_completions(self, choices: list[str], correct_index: int | None = None) -> list[str] | None:
        return []

    @override
    def get_cue_text(self) -> str:
        return ""


class _FakeMetric(BaseMetric[Any]):
    NAME = "FakeMetric"


class _StubTaskStyler(TaskStyler):
    """A stub styler whose declared metrics the metrics test reads back."""

    response_type = ResponseType.LOGLIKELIHOODS
    metrics: list[type[BaseMetric]] = [_FakeMetric]
    task_style = TaskStyle.MULTIPLE_CHOICE
    question_prefix = ""

    @override
    def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
        return ""

    @override
    def get_ground_truth(self, choices: list[str], correct_index: int) -> str:
        return ""

    @override
    def get_possible_completions(self, choices: list[str], correct_index: int | None = None) -> list[str] | None:
        return None

    @override
    def get_cue_text(self) -> str:
        return ""


def _make_benchmark(
    *,
    id: str = "dummy-benchmark",
    display_name: str | None = None,
    styler: TaskStyler | None = None,
    reader: ChoiceReader = _DUMMY_READER,
    sample_split: str = _DUMMY_SPLIT,
    fewshot_split: str = _DUMMY_SPLIT,
    subjects: SubjectsSelector = _DUMMY_SELECTOR,
    dataset_policy: DatasetPolicy | None = None,
    language: LanguageSpec = None,
) -> ComposedBenchmark:
    """Build a ``ComposedBenchmark`` for tests, defaulting to dummies for every argument the test does not provide."""
    return ComposedBenchmark.compose(
        id=id,
        display_name=display_name,
        kind=Choice(reader=reader, styler=styler or _DummyStyler()),
        sample_split=sample_split,
        fewshot_split=fewshot_split,
        subjects=subjects,
        dataset_policy=dataset_policy or _DummyDatasetPolicy(),
        language=language,
    )


def _make_eval(
    *,
    display_name: str = "dummy-eval",
    num_fewshot: int = 0,
    reader: ChoiceReader = _DUMMY_READER,
    loader: DatasetLoader = _DUMMY_LOADER,
    styler: TaskStyler | None = None,
    sample_split: str = _DUMMY_SPLIT,
    fewshot_split: str = _DUMMY_SPLIT,
    subjects: Subjects = _DUMMY_EVAL_SUBJECTS,
    language: LanguageSpec = None,
    rnd: random.Random = _DUMMY_RNG,
) -> ComposedEval:
    """Build a ``ComposedEval`` for tests, defaulting to dummies for every argument the test does not provide."""
    return ComposedEval(
        num_fewshot,
        display_name=display_name,
        kind=Choice(reader=reader, styler=styler or _DummyStyler()),
        loader=loader,
        sample_split=sample_split,
        fewshot_split=fewshot_split,
        subjects=subjects,
        language=language,
        rnd=rnd,
    )


def test_create_forwards_custom_subjects_to_its_selector() -> None:
    # Given a selector that records how create invokes it (selection itself is tested in test_subjects.py)
    class _SpySelector(SubjectsSelector):
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        @override
        def select(self, tokens: list[str]) -> Subjects:
            self.calls.append(tokens)
            return (Subject(load_key=None, label="dummy"),)

    selector = _SpySelector()
    benchmark = _make_benchmark(subjects=selector)

    # When creating evals with a selection and without one
    benchmark.create(0, ["b"], None)
    benchmark.create(0, None, None)

    # Then create forwards the tokens verbatim, mapping "no selection" to the empty (all) list
    assert selector.calls == [["b"], []]


def test_create_resolves_loader_through_policy_with_revision_override() -> None:
    # Given a policy that spies on how create invokes it
    class _SpyPolicy(DatasetPolicy):
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        @override
        def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
            self.calls.append(custom_hf_revision)
            return _DUMMY_LOADER

        @override
        def documentation(self) -> str:
            return ""

    policy = _SpyPolicy()

    # When creating an eval with a revision override
    _make_benchmark(dataset_policy=policy).create(0, None, "custom-sha")

    # Then create forwards the override to the policy
    assert policy.calls == ["custom-sha"]


def test_markdown_doc_renders_policy_dataset_section_and_example() -> None:
    # Given a policy whose loader serves a fixture dataset (no network) and documents itself
    class _FixtureLoader(DatasetLoader):
        @override
        def load(self, name: str | None) -> DatasetDict:
            return DatasetDict({_DUMMY_SPLIT: Dataset.from_list([{"x": 1}, {"x": 2}])})

        @override
        def metadata(self) -> dict[str, str]:
            return {}

    class _FixturePolicy(DatasetPolicy):
        @override
        def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
            return _FixtureLoader()

        @override
        def documentation(self) -> str:
            return "FIXTURE DATASET DOC"

    # When rendering the benchmark's documentation
    doc = _make_benchmark(dataset_policy=_FixturePolicy()).markdown_doc([ConcatFormatter()])

    # Then the policy's dataset section and an example prompt both appear
    assert "## Dataset\n\nFIXTURE DATASET DOC" in doc
    assert "## Example prompt" in doc


def test_display_name_defaults_to_id() -> None:
    benchmark = _make_benchmark(id="the-id")
    assert benchmark.display_name() == "the-id"


def test_id_stays_on_benchmark_display_name_reaches_eval() -> None:
    benchmark = _make_benchmark(id="the-id", display_name="Nice Name")
    assert (benchmark.id(), benchmark.display_name()) == ("the-id", "Nice Name")

    # id is a Benchmark concept; only display_name reaches the eval
    task = benchmark.create(0, None, None)
    assert task.display_name() == "Nice Name"


def test_user_prompt_suffix_rejected() -> None:
    # Composed evals have no completion path, so create rejects a user prompt suffix
    with pytest.raises(ValueError, match="only supported for completion tasks"):
        _make_benchmark().create(0, None, None, user_prompt_suffix="/think_short")


def test_cli_user_prompt_suffix_parsing() -> None:
    with patch("sys.argv", ["run.py", "--user-prompt-suffix", "/think_short"]):
        args = parse_args()

    assert args.user_prompt_suffix == "/think_short"


def test_metrics_combine_styler_and_response_type_metrics() -> None:
    # A benchmark's metrics are its styler's own metrics plus the ones its response type requires.
    benchmark = _make_benchmark(styler=_StubTaskStyler())
    assert set(benchmark.metrics()) == {_FakeMetric, BytesLoglikelihood, SequencePositionsLoglikelihood}


def test_get_metadata_reports_dataset_path_from_loader() -> None:
    # Given an eval whose loader reports a dataset path
    class _LoaderStub(DatasetLoader):
        @override
        def load(self, name: str | None) -> DatasetDict:
            return DatasetDict()

        @override
        def metadata(self) -> dict[str, str]:
            return {"dataset_path": "some/dataset"}

    task = _make_eval(loader=_LoaderStub())

    # Then get_metadata reports that dataset path
    assert task.get_metadata()["dataset_path"] == "some/dataset"


def test_message_sampling() -> None:
    # Given a reader that reads item["question"], a styler that echoes it + a cue,
    class _Reader(ChoiceReader):
        @override
        def read(self, item: dict[str, Any]) -> ChoiceFields:
            return ChoiceFields(raw_question=item["question"], choices=[], correct_index=0)

    class _Styler(_DummyStyler):
        @override
        def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
            return f"instruction: {raw_question}"

        @override
        def get_cue_text(self) -> str:
            return "the cue"

    # and a loader serving one row in the sample split
    class _Loader(DatasetLoader):
        @override
        def load(self, name: str | None) -> DatasetDict:
            return DatasetDict({_DUMMY_SPLIT: Dataset.from_list([{"question": "the goal"}])})

        @override
        def metadata(self) -> dict[str, str]:
            return {}

    task = _make_eval(reader=_Reader(), styler=_Styler(), loader=_Loader())

    # When iterating samples, then the instruction is the evaluated USER turn and the cue the ASSISTANT turn
    [sample] = list(task.iterate_samples())
    assert sample.messages == [
        Message(role=Role.USER, content="instruction: the goal"),
        Message(role=Role.ASSISTANT, content="the cue"),
    ]
