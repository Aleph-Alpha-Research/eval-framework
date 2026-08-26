import random
from typing import Any
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from eval_framework.composed import ChoiceFields, ChoiceReader, ComposedBenchmark, ComposedEval, LanguageSpec
from eval_framework.contract import ResponseType
from eval_framework.metrics.base import BaseMetric
from eval_framework.metrics.efficiency.bytes_per_sequence_position import (
    BytesLoglikelihood,
    SequencePositionsLoglikelihood,
)
from eval_framework.run import parse_args
from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy
from eval_framework.tasks.task_style import TaskStyle, TaskStyler
from template_formatting.formatter import ConcatFormatter, Message, Role


class _DummyReader(ChoiceReader):
    """A dummy reader: passed only to satisfy construction for doubles that never read (they override
    the styling methods, or raise before styling)."""

    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(raw_question="", choices=[], correct_index=0)


class _DummyDatasetLoader(DatasetLoader):
    """A no-op loader for doubles that never load a dataset."""

    def load(self, name: str | None) -> DatasetDict:
        return DatasetDict()

    def metadata(self) -> dict[str, str]:
        return {}


class _DummyDatasetPolicy(DatasetPolicy):
    """A no-op policy for benchmark doubles that never load a dataset."""

    def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
        return _DUMMY_LOADER

    def documentation(self) -> str:
        return ""


_DUMMY_READER = _DummyReader()
_DUMMY_LOADER = _DummyDatasetLoader()
_DUMMY_RNG = random.Random(0)
_DUMMY_SPLIT = "test"
_DUMMY_SUBJECTS = ["subject"]


class _DummyStyler(TaskStyler):
    """A dummy styler for tests that need a ComposedEval with a (loglikelihood) styler but never render
    a prompt — e.g. asserting a user_prompt_suffix is rejected."""

    response_type = ResponseType.LOGLIKELIHOODS
    metrics: list[type[BaseMetric]] = []
    task_style = TaskStyle.MULTIPLE_CHOICE
    question_prefix = ""

    def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
        return ""

    def get_ground_truth(self, choices: list[str], correct_index: int) -> str:
        return ""

    def get_possible_completions(self, choices: list[str], correct_index: int | None = None) -> list[str] | None:
        return None

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

    def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
        return ""

    def get_ground_truth(self, choices: list[str], correct_index: int) -> str:
        return ""

    def get_possible_completions(self, choices: list[str], correct_index: int | None = None) -> list[str] | None:
        return None

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
    subjects: list[Any] = _DUMMY_SUBJECTS,
    dataset_policy: DatasetPolicy | None = None,
    language: LanguageSpec = None,
) -> ComposedBenchmark:
    """Build a ``ComposedBenchmark`` for tests, defaulting to dummies for every argument the test does not provide."""
    return ComposedBenchmark.compose(
        id=id,
        display_name=display_name,
        styler=styler or _DummyStyler(),
        reader=reader,
        sample_split=sample_split,
        fewshot_split=fewshot_split,
        subjects=subjects,
        dataset_policy=dataset_policy or _DummyDatasetPolicy(),
        language=language,
    )


def _make_eval(
    *,
    id: str = "dummy-eval",
    display_name: str = "dummy-eval",
    num_fewshot: int = 0,
    reader: ChoiceReader = _DUMMY_READER,
    loader: DatasetLoader = _DUMMY_LOADER,
    styler: TaskStyler | None = None,
    sample_split: str = _DUMMY_SPLIT,
    fewshot_split: str = _DUMMY_SPLIT,
    subjects: list[Any] = _DUMMY_SUBJECTS,
    language: LanguageSpec = None,
    rnd: random.Random = _DUMMY_RNG,
) -> ComposedEval:
    """Build a ``ComposedEval`` for tests, defaulting to dummies for every argument the test does not provide."""
    return ComposedEval(
        num_fewshot,
        id=id,
        display_name=display_name,
        reader=reader,
        loader=loader,
        styler=styler or _DummyStyler(),
        sample_split=sample_split,
        fewshot_split=fewshot_split,
        subjects=subjects,
        language=language,
        rnd=rnd,
    )


@pytest.mark.parametrize(
    "subjects,custom_subjects,expected",
    [
        (["subject1", "subject2"], [], ["subject1", "subject2"]),
        (["subject1", "subject2"], None, ["subject1", "subject2"]),
        (["subject1", "subject2", "subject3"], ["subject1", "subject3"], ["subject1", "subject3"]),
        # result follows SUBJECTS' declared order, not the CLI argument order, and dedupes repeats --
        # matches come from `accepted_subjects`, not from echoing `custom_subjects` back verbatim.
        (["subject1", "subject2", "subject3"], ["subject3", "subject1"], ["subject1", "subject3"]),
        (["subject1", "subject2"], ["subject1", "subject1"], ["subject1"]),
        # "*" matches any scalar subject too, consistent with "*" already being a wildcard position
        # within a tuple subject. Redundant with custom_subjects=None/[] for the top-level case, but
        # the matching function treats scalar and tuple subjects the same way, so this falls out for free.
        (["subject1", "subject2", "subject3"], ["*"], ["subject1", "subject2", "subject3"]),
        ([("EN_US", "topic1"), ("EN_US", "topic2"), ("DE_DE", "topic1")], ["EN_US,topic1"], [("EN_US", "topic1")]),
        (
            [("EN_US", "topic1"), ("EN_US", "topic2"), ("DE_DE", "topic1")],
            ["EN_US,*"],
            [("EN_US", "topic1"), ("EN_US", "topic2")],
        ),
        (
            [
                ("EN_US", "topic1", "subtopic1"),
                ("EN_US", "topic1", "subtopic2"),
                ("EN_US", "topic2", "subtopic1"),
                ("DE_DE", "topic1", "subtopic1"),
            ],
            ["EN_US,topic1,*"],
            [("EN_US", "topic1", "subtopic1"), ("EN_US", "topic1", "subtopic2")],
        ),
        (
            [
                ("EN_US", "topic1", "subtopic1"),
                ("EN_US", "topic1", "subtopic2"),
                ("EN_US", "topic2", "subtopic1"),
                ("DE_DE", "topic1", "subtopic1"),
            ],
            ["*,topic1,*"],
            [
                ("EN_US", "topic1", "subtopic1"),
                ("EN_US", "topic1", "subtopic2"),
                ("DE_DE", "topic1", "subtopic1"),
            ],
        ),
        (
            [("EN_US", "topic1"), ("EN_US", "topic2"), ("DE_DE", "topic1")],
            ["EN_US,topic1", "DE_DE,topic1"],
            [("EN_US", "topic1"), ("DE_DE", "topic1")],
        ),
        # mixed-type tuple subjects, tuple[str, int, str]
        (
            [("ctx1", 4096, "single"), ("ctx1", 8192, "multi"), ("ctx2", 4096, "single")],
            ["ctx1,4096,single"],
            [("ctx1", 4096, "single")],
        ),
        (
            [("ctx1", 4096, "single"), ("ctx1", 8192, "multi"), ("ctx2", 4096, "single")],
            ["ctx1,*,*"],
            [("ctx1", 4096, "single"), ("ctx1", 8192, "multi")],
        ),
    ],
)
def test_task_custom_subjects(
    subjects: list[str] | list[tuple],
    custom_subjects: list[str] | None,
    expected: list[str] | list[tuple],
) -> None:
    # Filtering by custom subjects happens in create().
    task = _make_benchmark(subjects=subjects).create(0, custom_subjects, None)
    assert task.subjects == expected


@pytest.mark.parametrize(
    "subjects,custom_subjects",
    [
        (["subject1", "subject2"], ["invalid_subject"]),
        ([("EN_US", "topic1"), ("EN_US", "topic2")], ["EN_US,invalid_topic"]),
        ([("ctx1", 4096, "single"), ("ctx1", 8192, "multi")], ["ctx1,9999,single"]),
        # matching compares stringified subject fields, not a parsed native value, so a non-numeric
        # part at an int position is just another "not a legal value" case, not a separate parse error.
        ([("ctx1", 4096, "single"), ("ctx1", 8192, "multi")], ["ctx1,abc,single"]),
    ],
)
def test_task_custom_subjects_rejects_unknown(subjects: list[str] | list[tuple], custom_subjects: list[str]) -> None:
    with pytest.raises(ValueError):
        _make_benchmark(subjects=subjects).create(0, custom_subjects, None)


def test_create_resolves_loader_through_policy_with_revision_override() -> None:
    # Given a policy that spies on how create invokes it
    class _SpyPolicy(DatasetPolicy):
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
            self.calls.append(custom_hf_revision)
            return _DUMMY_LOADER

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
        def load(self, name: str | None) -> DatasetDict:
            return DatasetDict({_DUMMY_SPLIT: Dataset.from_list([{"x": 1}, {"x": 2}])})

        def metadata(self) -> dict[str, str]:
            return {}

    class _FixturePolicy(DatasetPolicy):
        def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
            return _FixtureLoader()

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


def test_id_and_display_name_reach_the_eval() -> None:
    benchmark = _make_benchmark(id="the-id", display_name="Nice Name")
    assert (benchmark.id(), benchmark.display_name()) == ("the-id", "Nice Name")

    task = benchmark.create(0, None, None)
    assert (task.id(), task.display_name()) == ("the-id", "Nice Name")


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
        def load(self, name: str | None) -> DatasetDict:
            return DatasetDict()

        def metadata(self) -> dict[str, str]:
            return {"dataset_path": "some/dataset"}

    task = _make_eval(loader=_LoaderStub())

    # Then get_metadata reports that dataset path
    assert task.get_metadata()["dataset_path"] == "some/dataset"


def test_message_sampling() -> None:
    # Given a reader that reads item["question"], a styler that echoes it + a cue,
    class _Reader(ChoiceReader):
        def read(self, item: dict[str, Any]) -> ChoiceFields:
            return ChoiceFields(raw_question=item["question"], choices=[], correct_index=0)

    class _Styler(_DummyStyler):
        def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
            return f"instruction: {raw_question}"

        def get_cue_text(self) -> str:
            return "the cue"

    # and a loader serving one row in the sample split
    class _Loader(DatasetLoader):
        def load(self, name: str | None) -> DatasetDict:
            return DatasetDict({_DUMMY_SPLIT: Dataset.from_list([{"question": "the goal"}])})

        def metadata(self) -> dict[str, str]:
            return {}

    task = _make_eval(reader=_Reader(), styler=_Styler(), loader=_Loader())

    # When iterating samples, then the instruction is the evaluated USER turn and the cue the ASSISTANT turn
    [sample] = list(task.iterate_samples())
    assert sample.messages == [
        Message(role=Role.USER, content="instruction: the goal"),
        Message(role=Role.ASSISTANT, content="the cue"),
    ]
