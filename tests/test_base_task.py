from typing import Any

import pytest

from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.registry import register_task
from tests.tasks.test_registry import temporary_registry


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
    self, subjects: list[str], custom_subjects: list[str] | None, expected_value: list[str]
) -> None:
    class MyTask(BaseTask):
        SUBJECTS = subjects
        NAME = "MyTask"

        def _get_instruction_text(self, item: dict[str, Any]) -> str:
            return ""

        def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
            return []

    register_task(MyTask)  # type: ignore[type-abstract]
    task = MyTask(num_fewshot=0, custom_subjects=custom_subjects, custom_hf_revision=None)
    if expected_value == "AssertionError":
        with pytest.raises(AssertionError):
            result = task.SUBJECTS
        return
    else:
        result = task.SUBJECTS
        assert result == expected_value
