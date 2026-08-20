from pathlib import Path

from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_loader import load_extra_tasks, load_modules_from_directory

TASK1 = """\
from eval_framework.tasks.registry import Registry
from eval_framework.contract import Benchmark


class DummyBenchmark(Benchmark):
    def id(self):
        return "MyCustomTask"

    def display_name(self):
        return "MyCustomTask"

    def response_type(self):
        raise NotImplementedError

    def metrics(self):
        raise NotImplementedError

    def subjects(self):
        raise NotImplementedError

    def create(self, num_fewshot, custom_subjects, custom_hf_revision, user_prompt_suffix=None, seed=None):
        raise NotImplementedError

    def markdown_doc(self, formatters):
        raise NotImplementedError


def register_tasks(registry: Registry) -> None:
    registry.add(DummyBenchmark())
"""

TASK2 = """\
from eval_framework.tasks.registry import Registry
from eval_framework.contract import Benchmark


class DummyBenchmark(Benchmark):
    def id(self):
        return "MySecondCustomTask"

    def display_name(self):
        return "MySecondCustomTask"

    def response_type(self):
        raise NotImplementedError

    def metrics(self):
        raise NotImplementedError

    def subjects(self):
        raise NotImplementedError

    def create(self, num_fewshot, custom_subjects, custom_hf_revision, user_prompt_suffix=None, seed=None):
        raise NotImplementedError

    def markdown_doc(self, formatters):
        raise NotImplementedError


def register_tasks(registry: Registry) -> None:
    registry.add(DummyBenchmark())
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
