from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from eval_framework.run import parse_args
from eval_framework.tasks import dataset_revisions as dr
from eval_framework.tasks.base import BaseTask, ResponseType
from template_formatting.formatter import Message, Role


@pytest.mark.parametrize(
    "subjects,custom_subjects,expected_value",
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
        (["subject1", "subject2"], ["invalid_subject"], "ValueError"),
        ([("EN_US", "topic1"), ("EN_US", "topic2")], ["EN_US,invalid_topic"], "ValueError"),
        ([("ctx1", 4096, "single"), ("ctx1", 8192, "multi")], ["ctx1,9999,single"], "ValueError"),
        # matching compares stringified subject fields, not a parsed native value, so a non-numeric
        # part at an int position is just another "not a legal value" case, not a separate parse error.
        ([("ctx1", 4096, "single"), ("ctx1", 8192, "multi")], ["ctx1,abc,single"], "ValueError"),
    ],
)
def test_task_custom_subjects(
    subjects: list[str] | list[tuple], custom_subjects: list[str] | None, expected_value: list[str] | list[tuple] | str
) -> None:
    class MyTask(BaseTask):
        REVISION_LOCKFILE = None
        SUBJECTS = subjects
        NAME = "MyTask"

        def _get_instruction_text(self, item: dict[str, Any]) -> str:
            return ""

        def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
            return []

    if expected_value == "ValueError":
        with pytest.raises(ValueError):
            task = MyTask.with_overwrite(num_fewshot=0, custom_subjects=custom_subjects, custom_hf_revision=None)
    else:
        task = MyTask.with_overwrite(num_fewshot=0, custom_subjects=custom_subjects, custom_hf_revision=None)
        result = task.SUBJECTS
        assert result == expected_value


def test_base_task() -> None:
    class MyTask1(BaseTask):
        REVISION_LOCKFILE = None
        NAME = "MyTask1"

        def _get_instruction_text(self, item: dict[str, Any]) -> str:
            return ""

        def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
            return []

    class MyTask2(BaseTask):
        REVISION_LOCKFILE = None
        NAME = "MyTask2"

        def _get_instruction_text(self, item: dict[str, Any]) -> str:
            return ""

        def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
            return []

    task1 = MyTask1()
    assert task1.NAME == "MyTask1"

    task2 = MyTask2.with_overwrite(0, custom_subjects=None, custom_hf_revision=None)
    assert task2.NAME == "MyTask2"


def test_user_prompt_suffix_only_applies_to_evaluated_user_turn() -> None:
    class MyTask(BaseTask):
        RESPONSE_TYPE = ResponseType.COMPLETION
        REVISION_LOCKFILE = None

        def _get_example_messages(self, item: dict[str, Any]) -> list[Message]:
            return [
                Message(role=Role.USER, content="fewshot question"),
                Message(role=Role.ASSISTANT, content="fewshot answer"),
            ]

        def _get_instruction_messages(self, item: dict[str, Any]) -> list[Message]:
            return [
                Message(role=Role.SYSTEM, content="instruction context"),
                Message(role=Role.USER, content="evaluated question"),
                Message(role=Role.ASSISTANT, content="intermediate cue"),
            ]

    task = MyTask.with_overwrite(
        1,
        custom_subjects=None,
        custom_hf_revision=None,
        user_prompt_suffix="/think_short",
    )

    messages = task._get_messages({})

    assert [message.content for message in messages] == [
        "fewshot question",
        "fewshot answer",
        "instruction context",
        "evaluated question/think_short",
        "intermediate cue",
    ]


def test_user_prompt_suffix_rejected_for_loglikelihood_task() -> None:
    class MyTask(BaseTask):
        RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
        REVISION_LOCKFILE = None

    with pytest.raises(ValueError, match="only supported for completion tasks"):
        MyTask.with_overwrite(
            0,
            custom_subjects=None,
            custom_hf_revision=None,
            user_prompt_suffix="/think_short",
        )


def test_cli_user_prompt_suffix_parsing() -> None:
    with patch("sys.argv", ["run.py", "--user-prompt-suffix", "/think_short"]):
        args = parse_args()

    assert args.user_prompt_suffix == "/think_short"


def _pinned_task(lockfile: Path | None) -> type[BaseTask]:
    """Test double declaring its own revision lock file, like any real task would."""

    class PinnedTask(BaseTask):
        NAME = "PinnedTask"
        DATASET_PATH = "my/dataset"
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
    task = _pinned_task(lockfile).with_overwrite(0, custom_subjects=None, custom_hf_revision=None)

    # Then the pinned revision is applied
    assert task.hf_revision == "pinned-sha"


def test_task_without_lockfile_is_not_pinned() -> None:
    # Given a task that opted out of pinning, when constructing it
    task = _pinned_task(None).with_overwrite(0, custom_subjects=None, custom_hf_revision=None)

    # Then no revision is pinned
    assert task.hf_revision is None


def test_missing_pin_in_declared_lockfile_raises(tmp_path: Path) -> None:
    # Given a task whose lock file has no pin for its dataset
    lockfile = tmp_path / "hf-dataset-revisions.json"
    dr.HfDatasetRevisions({}).to_file(lockfile)

    # Then constructing the task fails
    with pytest.raises(KeyError, match="not pinned"):
        _pinned_task(lockfile).with_overwrite(0, custom_subjects=None, custom_hf_revision=None)


def test_custom_hf_revision_overrides_pinned(tmp_path: Path) -> None:
    # Given a task whose lock file pins its dataset
    lockfile = tmp_path / "hf-dataset-revisions.json"
    dr.HfDatasetRevisions({"my/dataset": "pinned-sha"}).to_file(lockfile)

    # When constructing the task with a revision override
    task = _pinned_task(lockfile).with_overwrite(0, custom_subjects=None, custom_hf_revision="custom-sha")

    # Then the override beats the pin
    assert task.hf_revision == "custom-sha"
