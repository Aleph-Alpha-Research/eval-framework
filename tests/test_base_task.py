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


def test_base_task() -> None:
    class MyTask1(BaseTask):
        NAME = "MyTask2"

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
