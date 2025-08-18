import os
import tempfile
import textwrap

import pytest

from eval_framework.task_loader import load_extra_tasks
from eval_framework.task_names import TaskName
from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.eval_config import EvalConfig
from tests.conftest import MockLLM


def test_user_task_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_file = os.path.join(tmpdir, "my_custom_task.py")
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
        monkeypatch.syspath_prepend(tmpdir)
        load_extra_tasks([task_file])
        assert hasattr(TaskName, "MyCustomTask".upper())
        task_cls = getattr(TaskName, "MyCustomTask".upper()).value
        assert issubclass(task_cls, BaseTask)
        assert task_cls.NAME == "MyCustomTask"
        assert TaskName.from_name("MyCustomTask")


def test_user_task_registration_plus_builtin_task(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_file = os.path.join(tmpdir, "my_second_custom_task.py")
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
        monkeypatch.syspath_prepend(tmpdir)
        load_extra_tasks([task_file])
        assert TaskName.from_name("MySecondCustomTask")  # Ensure it can be accessed by name
        assert hasattr(TaskName, "MySecondCustomTask".upper())
        task_cls = getattr(TaskName, "MySecondCustomTask".upper()).value
        assert issubclass(task_cls, BaseTask)
        assert task_cls.NAME == "MySecondCustomTask"
        assert TaskName.from_name("MySecondCustomTask")


def test_user_task_registration_with_EvalConfig(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_file = os.path.join(tmpdir, "my_second_custom_task.py")
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
        monkeypatch.syspath_prepend(tmpdir)
        monkeypatch.syspath_prepend(tmpdir)
        load_extra_tasks([task_file])
        config = EvalConfig(task_name="MyThirdCustomTask", llm_class=MockLLM)
        assert config.task_name.value.NAME == "MyThirdCustomTask"
        assert TaskName.from_name("MyThirdCustomTask")


def test_derived_user_task_registration(monkeypatch: pytest.MonkeyPatch) -> None:
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
