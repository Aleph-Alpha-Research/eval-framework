import importlib.util
import inspect
import logging
import os
from types import ModuleType
from typing import Sequence

from aenum import extend_enum

from eval_framework.task_names import TaskName
from eval_framework.tasks.base import BaseTask

logger = logging.getLogger(__name__)


def load_extra_tasks(module_paths: Sequence[str]) -> None:
    """
    Dynamically load and register user-defined tasks from a list of files or directories.
    Each .py file found will be imported, and any BaseTask subclass will be registered
    in the TaskName enum for use by name.
    Provides clear error messages for missing/invalid files or import errors.
    """
    py_files = []
    for path in module_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[User Task Loader] Path does not exist: {path}")
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".py") and not file.startswith("__"):
                        py_files.append(os.path.join(root, file))
        elif os.path.isfile(path) and path.endswith(".py"):
            py_files.append(path)
        else:
            raise ValueError(f"[User Task Loader] Path is not a .py file or directory: {path}")

    for file_path in py_files:
        try:
            spec = importlib.util.spec_from_file_location("user_task_module", file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"[User Task Loader] Could not create a module spec for {file_path}")
            user_module: ModuleType = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_module)
        except Exception as e:
            raise ImportError(f"[User Task Loader] Failed to import {file_path}: {e}") from e

        for name, obj in inspect.getmembers(user_module):
            if inspect.isclass(obj) and issubclass(obj, BaseTask) and obj is not BaseTask:
                name_upper = obj.NAME.upper()
                if hasattr(TaskName, obj.NAME) or hasattr(TaskName, name_upper):
                    logger.info(obj.__module__)

                    if "eval_framework.tasks.benchmarks" not in obj.__module__:
                        # skip if import comes from eval_framework's built-in tasks
                        raise ValueError(f"Duplicate user task name found (case-insensitive): {obj.NAME}")
                else:
                    # setattr(TaskName, name_upper, obj)
                    class_obj = getattr(user_module, name)
                    extend_enum(TaskName, name_upper, class_obj)
                    logger.info(f"[User Task Loader] Registered task: {name_upper}")
