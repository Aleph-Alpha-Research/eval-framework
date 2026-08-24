from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from eval_framework.composed import ChoiceFields, ChoiceReader, ComposedBenchmark, ComposedEval
from eval_framework.contract import ResponseType
from eval_framework.metrics.base import BaseMetric
from eval_framework.metrics.efficiency.bytes_per_sequence_position import (
    BytesLoglikelihood,
    SequencePositionsLoglikelihood,
)
from eval_framework.run import parse_args
from eval_framework.tasks import dataset_revisions as dr
from eval_framework.tasks.task_style import TaskStyle, TaskStyler
from template_formatting.formatter import Message, Role


class _DummyReader(ChoiceReader):
    """A dummy reader: passed only to satisfy construction for doubles that never read (they override
    the styling methods, or raise before styling)."""

    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(raw_question="", choices=[], correct_index=0)


_DUMMY_READER = _DummyReader()
_DUMMY_DATASET_PATH = "dummy/dataset"
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


def _benchmark_with_subjects(subjects: list[Any]) -> ComposedBenchmark:
    class MyTask(ComposedEval):
        REVISION_LOCKFILE = None
        NAME = "MyTask"
        TASK_STYLER = _DummyStyler()

    return ComposedBenchmark.from_base(
        MyTask,
        reader=_DUMMY_READER,
        dataset_path=_DUMMY_DATASET_PATH,
        sample_split=_DUMMY_SPLIT,
        fewshot_split=_DUMMY_SPLIT,
        subjects=subjects,
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
    subjects: list[str] | list[tuple], custom_subjects: list[str] | None, expected: list[str] | list[tuple]
) -> None:
    # Filtering by custom subjects happens in create().
    task = _benchmark_with_subjects(subjects).create(0, custom_subjects, None)
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
        _benchmark_with_subjects(subjects).create(0, custom_subjects, None)


def test_base_task() -> None:
    class MyTask1(ComposedEval):
        REVISION_LOCKFILE = None
        NAME = "MyTask1"

        def _get_instruction_text(self, item: dict[str, Any]) -> str:
            return ""

        def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
            return []

    class MyTask2(ComposedEval):
        REVISION_LOCKFILE = None
        NAME = "MyTask2"

        def _get_instruction_text(self, item: dict[str, Any]) -> str:
            return ""

        def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
            return []

    task1 = MyTask1(
        reader=_DUMMY_READER,
        dataset_path=_DUMMY_DATASET_PATH,
        sample_split=_DUMMY_SPLIT,
        fewshot_split=_DUMMY_SPLIT,
        subjects=_DUMMY_SUBJECTS,
    )
    assert task1.NAME == "MyTask1"

    task2 = MyTask2(
        0,
        reader=_DUMMY_READER,
        dataset_path=_DUMMY_DATASET_PATH,
        sample_split=_DUMMY_SPLIT,
        fewshot_split=_DUMMY_SPLIT,
        subjects=_DUMMY_SUBJECTS,
        custom_hf_revision=None,
    )
    assert task2.NAME == "MyTask2"


def test_user_prompt_suffix_rejected() -> None:
    # Given a composed eval (composed evals have no completion path, so a suffix is never valid)
    class MyTask(ComposedEval):
        REVISION_LOCKFILE = None
        TASK_STYLER = _DummyStyler()

    # When constructing it with a user prompt suffix, then it is rejected
    with pytest.raises(ValueError, match="only supported for completion tasks"):
        MyTask(
            0,
            reader=_DUMMY_READER,
            dataset_path=_DUMMY_DATASET_PATH,
            sample_split=_DUMMY_SPLIT,
            fewshot_split=_DUMMY_SPLIT,
            subjects=_DUMMY_SUBJECTS,
            custom_hf_revision=None,
            user_prompt_suffix="/think_short",
        )


def test_cli_user_prompt_suffix_parsing() -> None:
    with patch("sys.argv", ["run.py", "--user-prompt-suffix", "/think_short"]):
        args = parse_args()

    assert args.user_prompt_suffix == "/think_short"


def _pinned_task(lockfile: Path | None) -> type[ComposedEval]:
    """Test double declaring its own revision lock file, like any real task would."""

    class PinnedTask(ComposedEval):
        NAME = "PinnedTask"
        REVISION_LOCKFILE = lockfile

        def _get_instruction_text(self, item: dict[str, Any]) -> str:
            return ""

        def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
            return []

    return PinnedTask


def test_pinned_hf_revision_applied_when_unset(tmp_path: Path) -> None:
    # Given a task whose lock file pins its dataset
    lockfile = tmp_path / "hf-dataset-revisions.json"
    dr.HfDatasetRevisions({"my/dataset": "pinned-sha"}).to_file(lockfile)

    # When constructing the task without a revision override
    task = _pinned_task(lockfile)(
        0,
        reader=_DUMMY_READER,
        dataset_path="my/dataset",
        sample_split=_DUMMY_SPLIT,
        fewshot_split=_DUMMY_SPLIT,
        subjects=_DUMMY_SUBJECTS,
        custom_hf_revision=None,
    )

    # Then the pinned revision is applied
    assert task.hf_revision == "pinned-sha"


def test_task_without_lockfile_is_not_pinned() -> None:
    # Given a task that opted out of pinning, when constructing it
    task = _pinned_task(None)(
        0,
        reader=_DUMMY_READER,
        dataset_path="my/dataset",
        sample_split=_DUMMY_SPLIT,
        fewshot_split=_DUMMY_SPLIT,
        subjects=_DUMMY_SUBJECTS,
        custom_hf_revision=None,
    )

    # Then no revision is pinned
    assert task.hf_revision is None


def test_missing_pin_in_declared_lockfile_raises(tmp_path: Path) -> None:
    # Given a task whose lock file has no pin for its dataset
    lockfile = tmp_path / "hf-dataset-revisions.json"
    dr.HfDatasetRevisions({}).to_file(lockfile)

    # Then constructing the task fails
    with pytest.raises(KeyError, match="not pinned"):
        _pinned_task(lockfile)(
            0,
            reader=_DUMMY_READER,
            dataset_path="my/dataset",
            sample_split=_DUMMY_SPLIT,
            fewshot_split=_DUMMY_SPLIT,
            subjects=_DUMMY_SUBJECTS,
            custom_hf_revision=None,
        )


def test_custom_hf_revision_overrides_pinned(tmp_path: Path) -> None:
    # Given a task whose lock file pins its dataset
    lockfile = tmp_path / "hf-dataset-revisions.json"
    dr.HfDatasetRevisions({"my/dataset": "pinned-sha"}).to_file(lockfile)

    # When constructing the task with a revision override
    task = _pinned_task(lockfile)(
        0,
        reader=_DUMMY_READER,
        dataset_path="my/dataset",
        sample_split=_DUMMY_SPLIT,
        fewshot_split=_DUMMY_SPLIT,
        subjects=_DUMMY_SUBJECTS,
        custom_hf_revision="custom-sha",
    )

    # Then the override beats the pin
    assert task.hf_revision == "custom-sha"


def test_get_metrics_combines_styler_and_response_type_metrics() -> None:
    # Given a composed eval whose styler declares its own metrics
    class MyTask(ComposedEval):
        REVISION_LOCKFILE = None
        NAME = "MyTask"
        TASK_STYLER = _StubTaskStyler()

    # Then its metrics are the styler's metrics plus the loglikelihood response-type metrics
    task = MyTask(
        reader=_DUMMY_READER,
        dataset_path=_DUMMY_DATASET_PATH,
        sample_split=_DUMMY_SPLIT,
        fewshot_split=_DUMMY_SPLIT,
        subjects=_DUMMY_SUBJECTS,
    )
    assert set(task.get_metrics()) == {_FakeMetric, BytesLoglikelihood, SequencePositionsLoglikelihood}


def test_get_messages_assembles_instruction_and_cue() -> None:
    # Given a reader that reads the question from item["question"], and a styler that echoes it + a cue
    class _Reader(ChoiceReader):
        def read(self, item: dict[str, Any]) -> ChoiceFields:
            return ChoiceFields(raw_question=item["question"], choices=[], correct_index=0)

    class _Styler(_DummyStyler):
        def get_instruction_text(self, raw_question: str, choices: list[str]) -> str:
            return f"instruction: {raw_question}"

        def get_cue_text(self) -> str:
            return "the cue"

    class MyTask(ComposedEval):
        REVISION_LOCKFILE = None
        NAME = "MyTask"
        TASK_STYLER = _Styler()

    task = MyTask(
        reader=_Reader(),
        dataset_path=_DUMMY_DATASET_PATH,
        sample_split=_DUMMY_SPLIT,
        fewshot_split=_DUMMY_SPLIT,
        subjects=_DUMMY_SUBJECTS,
    )

    # Then the instruction becomes the evaluated USER turn and the cue an ASSISTANT turn
    assert task._get_messages({"question": "the goal"}) == [
        Message(role=Role.USER, content="instruction: the goal"),
        Message(role=Role.ASSISTANT, content="the cue"),
    ]
