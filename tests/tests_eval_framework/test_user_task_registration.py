from pathlib import Path

from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_loader import load_extra_tasks, load_modules_from_directory

TASK1 = """\
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.registry import Registry, register_task

class MyCustomTask(BaseTask):
    NAME = "MyCustomTask"
    DATASET_PATH = "dummy"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = []
    SUBJECTS = []
    LANGUAGE = Language.ENG

def register_tasks(registry: Registry) -> None:
    register_task(MyCustomTask, registry)
"""

TASK2 = """\
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.registry import Registry, register_task

class MySecondCustomTask(BaseTask):
    NAME = "MySecondCustomTask"
    DATASET_PATH = "dummy"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = []
    SUBJECTS = []
    LANGUAGE = Language.ENG

def register_tasks(registry: Registry) -> None:
    register_task(MySecondCustomTask, registry)
"""


def test_user_task_registration(tmp_path: Path) -> None:
    registry = Registry()
    (tmp_path / "my_custom_task.py").write_text(TASK1)

    load_extra_tasks(tmp_path, registry=registry)
    assert "MyCustomTask" in registry
    assert registry["MyCustomTask"].display_name() == "MyCustomTask"
    assert set(registry.task_names()) == {"MyCustomTask"}


def test_directory_recursive(tmp_path: Path) -> None:
    registry = Registry()
    subdir = tmp_path / "my_custom_tasks"
    subdir.mkdir()

    (tmp_path / "task1.py").write_text(TASK1)
    (subdir / "task2.py").write_text(TASK2)

    load_extra_tasks(tmp_path, registry=registry)
    assert set(registry.task_names()) == {"MyCustomTask", "MySecondCustomTask"}
    assert registry["MyCustomTask"] is not registry["MySecondCustomTask"]


def test_modules_without_register_tasks_are_ignored(tmp_path: Path) -> None:
    registry = Registry()
    (tmp_path / "task1.py").write_text(TASK1)
    # A helper module with no register_tasks entrypoint should simply be skipped.
    (tmp_path / "helper.py").write_text("VALUE = 42\n")

    load_extra_tasks(tmp_path, registry=registry)
    assert set(registry.task_names()) == {"MyCustomTask"}


def test_unique_module_names(tmp_path: Path) -> None:
    subdir = tmp_path / "nested"
    subdir.mkdir()
    (tmp_path / "task1.py").write_text(TASK1)
    (subdir / "task2.py").write_text(TASK2)

    modules = load_modules_from_directory(tmp_path)
    names = {module.__name__ for module in modules}
    assert names == {"user_tasks.task1", "user_tasks.nested.task2"}
