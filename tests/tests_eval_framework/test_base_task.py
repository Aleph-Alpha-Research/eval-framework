from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from eval_framework.run import parse_args
from eval_framework.tasks import dataset_revisions as dr
from eval_framework.tasks.base import BaseTask, ResponseType
from eval_framework.tasks.benchmarks.copa import COPA
from eval_framework.tasks.benchmarks.piqa import PIQA
from eval_framework.tasks.registry import register_task
from template_formatting.formatter import Message, Role
from tests.tests_eval_framework.tasks.conftest import FIXTURE_REVISIONS, write_fixture_revisions_file
from tests.tests_eval_framework.tasks.test_registry import temporary_registry


@pytest.mark.parametrize(
    "subjects,custom_subjects,expected_value",
    [
        (["subject1", "subject2"], [], ["subject1", "subject2"]),
        (["subject1", "subject2"], None, ["subject1", "subject2"]),
        (["subject1", "subject2", "subject3"], ["subject1", "subject3"], ["subject1", "subject3"]),
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
        (["subject1", "subject2"], ["invalid_subject"], "AssertionError"),
        ([("EN_US", "topic1"), ("EN_US", "topic2")], ["EN_US,invalid_topic"], "AssertionError"),
    ],
)
@temporary_registry
def test_task_custom_subjects(
    subjects: list[str] | list[tuple], custom_subjects: list[str] | None, expected_value: list[str] | list[tuple] | str
) -> None:
    class MyTask(BaseTask):
        SUBJECTS = subjects
        NAME = "MyTask"

        def _get_instruction_text(self, item: dict[str, Any]) -> str:
            return ""

        def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
            return []

    register_task(MyTask)  # type: ignore[type-abstract]
    if expected_value == "AssertionError":
        with pytest.raises(AssertionError):
            task = MyTask.with_overwrite(num_fewshot=0, custom_subjects=custom_subjects, custom_hf_revision=None)
    else:
        task = MyTask.with_overwrite(num_fewshot=0, custom_subjects=custom_subjects, custom_hf_revision=None)
        result = task.SUBJECTS
        assert result == expected_value


@temporary_registry
def test_base_task() -> None:
    class MyTask1(BaseTask):
        NAME = "MyTask1"

        def _get_instruction_text(self, item: dict[str, Any]) -> str:
            return ""

        def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
            return []

    class MyTask2(BaseTask):
        NAME = "MyTask2"

        def _get_instruction_text(self, item: dict[str, Any]) -> str:
            return ""

        def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
            return []

    register_task(MyTask1)  # type: ignore[type-abstract]
    task1 = MyTask1()
    assert task1.NAME == "MyTask1"

    register_task(MyTask2)  # type: ignore[type-abstract]
    task2 = MyTask2.with_overwrite(0, custom_subjects=None, custom_hf_revision=None)
    assert task2.NAME == "MyTask2"


def test_user_prompt_suffix_only_applies_to_evaluated_user_turn() -> None:
    class MyTask(BaseTask):
        RESPONSE_TYPE = ResponseType.COMPLETION

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


def test_pinned_hf_revision_applied_when_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture_path = write_fixture_revisions_file(tmp_path)
    dr._pinned_revisions.cache_clear()
    monkeypatch.setattr(dr.DatasetRevision, "_INSTANCE", None)
    dr.DatasetRevision.add_revision_file(fixture_path)
    task = COPA.with_overwrite(0, custom_subjects=None, custom_hf_revision=None)
    assert task.HF_REVISION == FIXTURE_REVISIONS["COPA"]


def test_custom_hf_revision_overrides_pinned() -> None:
    task = COPA.with_overwrite(0, custom_subjects=None, custom_hf_revision="custom-sha")
    assert task.HF_REVISION == "custom-sha"


def test_class_hf_revision_not_overridden_by_pin_file() -> None:
    task = PIQA.with_overwrite(0, custom_subjects=None, custom_hf_revision=None)
    assert task.HF_REVISION == "6b3aceb3276e5ab7e51895d73151a718690af38c"
