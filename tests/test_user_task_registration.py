import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from eval_framework.task_loader import load_extra_tasks
from eval_framework.task_names import TaskName
from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.eval_config import EvalConfig
from tests.conftest import MockLLM


def test_user_task_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_file = os.path.join(tmp_path, "my_custom_task.py")
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
    monkeypatch.syspath_prepend(tmp_path)
    load_extra_tasks([task_file])
    assert hasattr(TaskName, "MyCustomTask".upper())
    task_cls = getattr(TaskName, "MyCustomTask".upper()).value
    assert issubclass(task_cls, BaseTask)
    assert task_cls.NAME == "MyCustomTask"
    assert TaskName.from_name("MyCustomTask")


def test_user_task_registration_plus_builtin_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_file = os.path.join(tmp_path, "my_second_custom_task.py")
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
    monkeypatch.syspath_prepend(tmp_path)
    load_extra_tasks([task_file])
    assert TaskName.from_name("MySecondCustomTask")  # Ensure it can be accessed by name
    assert hasattr(TaskName, "MySecondCustomTask".upper())
    task_cls = getattr(TaskName, "MySecondCustomTask".upper()).value
    assert issubclass(task_cls, BaseTask)
    assert task_cls.NAME == "MySecondCustomTask"
    assert TaskName.from_name("MySecondCustomTask")


def test_user_task_registration_with_EvalConfig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_file = os.path.join(tmp_path, "my_second_custom_task.py")
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
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.syspath_prepend(tmp_path)
    load_extra_tasks([task_file])
    config = EvalConfig(task_name="MyThirdCustomTask", llm_class=MockLLM)
    assert config.task_name.value.NAME == "MyThirdCustomTask"
    assert TaskName.from_name("MyThirdCustomTask")


def test_derived_user_task_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_file = os.path.join(tmpdir, "my_derived_task.py")
        with open(task_file, "w") as f:
            f.write(
                textwrap.dedent("""
                from eval_framework.tasks.benchmarks.copa import COPA
                class MyCOPA(COPA):
                    NAME = "MyCOPA"
            """)
            )
        monkeypatch.syspath_prepend(tmpdir)
        load_extra_tasks([task_file])
        assert TaskName.from_name("MyCOPA")


def test_user_task_registration_with_repeated_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that loading user tasks with duplicate names raises an error."""
    task_file = os.path.join(tmp_path, "my_custom_task.py")
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
    monkeypatch.syspath_prepend(tmp_path)

    with pytest.raises(ValueError, match="Duplicate user task"):
        load_extra_tasks([task_file])
