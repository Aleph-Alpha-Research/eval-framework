import textwrap
from pathlib import Path

import pytest

from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.registry import get_task, is_registered
from eval_framework.tasks.task_loader import load_extra_tasks
from tests.tasks.test_registry import temporary_registry


@temporary_registry
def test_user_task_registration(tmp_path: Path) -> None:
    task_file = tmp_path / "my_custom_task.py"
    with open(task_file, "w") as f:
        f.write(
            textwrap.dedent("""
            from eval_framework.tasks.base import BaseTask, Language
            class MyCustomTask(BaseTask):
                NAME = "MyCustomTask"
                DATASET_PATH = "dummy"
                SAMPLE_SPLIT = "test"
                FEWSHOT_SPLIT = "test"
                RESPONSE_TYPE = None
                METRICS = []
                SUBJECTS = []
                LANGUAGE = Language.ENG
        """)
        )
    load_extra_tasks([task_file])
    assert is_registered("MyCustomTask")
    task_cls = get_task("MyCustomTask")
    assert issubclass(task_cls, BaseTask)
    assert task_cls.NAME == "MyCustomTask"


@temporary_registry
def test_user_task_registration_plus_builtin_task(tmp_path: Path) -> None:
    task_file = tmp_path / "my_second_custom_task.py"
    with open(task_file, "w") as f:
        f.write(
            textwrap.dedent("""
            from eval_framework.tasks.base import BaseTask, Language
            class MySecondCustomTask(BaseTask):
                NAME = "MySecondCustomTask"
                DATASET_PATH = "dummy"
                SAMPLE_SPLIT = "test"
                FEWSHOT_SPLIT = "test"
                RESPONSE_TYPE = None
                METRICS = []
                SUBJECTS = []
                LANGUAGE = Language.ENG
        """)
        )
    load_extra_tasks([task_file])
    assert is_registered("MySecondCustomTask")
    assert is_registered("MySecondCustomTask".upper())
    task_cls = get_task("MySecondCustomTask".upper())
    assert issubclass(task_cls, BaseTask)
    assert task_cls.NAME == "MySecondCustomTask"


@temporary_registry
def test_user_task_registration_with_EvalConfig(tmp_path: Path) -> None:
    task_file = tmp_path / "my_second_custom_task.py"
    with open(task_file, "w") as f:
        f.write(
            textwrap.dedent("""
            from eval_framework.tasks.base import BaseTask, Language
            class MyThirdCustomTask(BaseTask):
                NAME = "MyThirdCustomTask"
                DATASET_PATH = "dummy"
                SAMPLE_SPLIT = "test"
                FEWSHOT_SPLIT = "test"
                RESPONSE_TYPE = None
                METRICS = []
                SUBJECTS = []
                LANGUAGE = Language.ENG
        """)
        )
    load_extra_tasks([task_file])
    assert is_registered("MyThirdCustomTask")
    task = get_task("MyThirdCustomTask")
    assert task.DATASET_PATH == "dummy"


@temporary_registry
def test_derived_user_task_registration(tmp_path: Path) -> None:
    task_file = tmp_path / "my_derived_task.py"
    with open(task_file, "w") as f:
        f.write(
            textwrap.dedent("""
            from eval_framework.tasks.benchmarks.copa import COPA
            class MyCOPA(COPA):
                NAME = "MyCOPA"
        """)
        )
    load_extra_tasks([task_file])
    assert is_registered("MyCOPA")
    get_task("MyCOPA")


@temporary_registry
def test_user_task_registration_with_repeated_names(tmp_path: Path) -> None:
    """Test that loading user tasks with duplicate names raises an error."""
    task_file = tmp_path / "my_custom_task.py"
    with open(task_file, "w") as f:
        f.write(
            textwrap.dedent("""
            from eval_framework.tasks.base import BaseTask, Language
            class MyCustomTask(BaseTask):
                NAME = "MyCustomTask"
                DATASET_PATH = "dummy"
                SAMPLE_SPLIT = "test"
                FEWSHOT_SPLIT = "test"
                RESPONSE_TYPE = None
                METRICS = []
                SUBJECTS = []
                LANGUAGE = Language.ENG

            class MyCustomTask2(BaseTask):
                NAME = "MyCustomTask" # repeated name
                DATASET_PATH = "dummy"
                SAMPLE_SPLIT = "test"
                FEWSHOT_SPLIT = "test"
                RESPONSE_TYPE = None
                METRICS = []
                SUBJECTS = []
                LANGUAGE = Language.ENG
        """)
        )

    with pytest.raises(ValueError, match="Duplicate user task"):
        load_extra_tasks([task_file])
