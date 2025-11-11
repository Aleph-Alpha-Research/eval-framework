import textwrap
from pathlib import Path

import pytest

from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.registry import get_task, is_registered, registered_task_names
from eval_framework.tasks.task_loader import find_all_python_files, load_extra_tasks
from tests.tests_eval_framework.tasks.test_registry import temporary_registry

TASK1 = """\
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
"""

TASK2 = """\
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
"""


@temporary_registry
def test_user_task_registration(tmp_path: Path) -> None:
    task_file = tmp_path / "my_custom_task.py"
    with open(task_file, "w") as f:
        f.write(TASK1)
    load_extra_tasks([task_file])
    assert is_registered("MyCustomTask")
    task1 = get_task("MyCustomTask")
    assert issubclass(task1, BaseTask)
    assert task1.NAME == "MyCustomTask"
    assert set(registered_task_names()) == {"MyCustomTask"}

    task_file = tmp_path / "my_second_custom_task.py"
    with open(task_file, "w") as f:
        f.write(TASK2)
    load_extra_tasks([task_file])
    assert is_registered("MySecondCustomTask")
    assert is_registered("MySecondCustomTask".upper())
    task2 = get_task("MySecondCustomTask".upper())
    assert issubclass(task2, BaseTask)
    assert task2.NAME == "MySecondCustomTask"
    assert set(registered_task_names()) == {"MyCustomTask", "MySecondCustomTask"}

    assert task1 is not task2


@temporary_registry
def test_directory(tmp_path: Path) -> None:
    subdir = tmp_path / "my_custom_tasks"
    subdir.mkdir()

    with open(tmp_path / "task1.py", "w") as f:
        f.write(TASK1)
    with open(subdir / "task2.py", "w") as f:
        f.write(TASK2)

    load_extra_tasks([tmp_path])
    assert set(registered_task_names()) == {"MyCustomTask", "MySecondCustomTask"}
    assert get_task("MyCustomTask") != get_task("MySecondCustomTask")


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


def test_find_all_python_files(tmp_path: Path) -> None:
    subdir = tmp_path / "dir1" / "dir2"
    subdir.mkdir(parents=True)

    file1 = tmp_path / "file.py"
    file2 = subdir / "file.py"

    file1.touch()
    file2.touch()

    assert find_all_python_files(file1) == {file1}
    assert find_all_python_files(subdir) == {file2}
    assert find_all_python_files(tmp_path) == {file1, file2}
    # Overlapping paths (duplicates)
    assert find_all_python_files(tmp_path, subdir) == {file1, file2}
    # File / directory mixture
    assert find_all_python_files(file1, subdir) == {file1, file2}
