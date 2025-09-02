import importlib.util
import inspect
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.registry import is_registered, register_task

logger = logging.getLogger(__name__)


def iter_python_files(*module_paths: str | os.PathLike) -> Iterator[Path]:
    """Recursively walk through all paths and yield Python files."""
    for path in module_paths:
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"[User Task Loader] Path does not exist: {path}")
        if path.is_dir():
            yield from path.glob("**/*.py")
        elif path.is_file():
            if path.suffix != ".py":
                raise ValueError(f"The provided path {path} is not a Python file.")
            yield path
        else:
            raise ValueError(f"Path is not a .py file or directory: {path}")


def import_file(f: str | os.PathLike, /) -> Any:
    """Import a file as a Python module."""
    file_path = Path(f)
    try:
        spec = importlib.util.spec_from_file_location("user_task_module", file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create a module spec for {file_path}")
        user_module: ModuleType = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_module)
    except Exception as e:
        raise ImportError(f"Failed to import {file_path}: {e}") from e
    return user_module


def load_extra_tasks(module_paths: Sequence[str | os.PathLike]) -> None:
    """Dynamically load and register user-defined tasks from a list of files or directories.

    Each .py file found will be imported, and any BaseTask subclass will be registered
    in the TaskName enum for use by name.
    Provides clear error messages for missing/invalid files or import errors.
    """
    for file_path in iter_python_files(*module_paths):
        user_module = import_file(file_path)

        for name, obj in inspect.getmembers(user_module):
            if inspect.isclass(obj) and issubclass(obj, BaseTask) and obj is not BaseTask:
                # check unique name of BaseTask subclasses (when they have a NAME attribute)
                if not hasattr(obj, "NAME"):
                    logger.info(f"[User Task Loader] Skipping {obj.__module__} - no NAME attribute present.")
                else:
                    if is_registered(obj.NAME):
                        # two classes with the same NAME attribute
                        logger.info(obj.__module__)

                        # if it comes from eval_framework's built-in tasks then will just skip (no need to register
                        # again; this can happen if a class is imported so that a new task can be derived from it)
                        # but if it is one of the user defined task then raise a duplicated name error
                        if "eval_framework.tasks.benchmarks" not in obj.__module__:
                            # skip if import comes from eval_framework's built-in tasks
                            raise ValueError(f"Duplicate user task name found (case-insensitive): {obj.NAME}")

                    else:
                        # if there is no duplicate name conflict then register the new task
                        class_obj = getattr(user_module, name)
                        register_task(class_obj.NAME, class_obj)
                        logger.info(f"[User Task Loader] Registered task: {class_obj.NAME}")
