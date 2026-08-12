import importlib.util
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from eval_framework.tasks.registry import Registry
from eval_framework.tasks.registry import registry as global_registry

logger = logging.getLogger(__name__)

# Prefix used to namespace dynamically loaded user plugin modules in ``sys.modules``.
USER_PLUGIN_NAMESPACE = "user_tasks"

# Name of the entrypoint function each plugin module may define to register tasks.
REGISTER_TASKS_ENTRYPOINT = "register_tasks"


def _module_name_for(file_path: Path, root: Path) -> str:
    """Derive a unique, importable module name from a plugin file's path.

    The name is namespaced under ``USER_PLUGIN_NAMESPACE`` and mirrors the file's
    location relative to the plugin root, e.g. ``<root>/subdir/foo.py`` becomes
    ``user_tasks.subdir.foo``.
    """
    relative = file_path.relative_to(root).with_suffix("")
    return ".".join((USER_PLUGIN_NAMESPACE, *relative.parts))


def load_modules_from_directory(directory: str | os.PathLike) -> list[ModuleType]:
    """Recursively import every ``.py`` file under ``directory`` as a module.

    Each file is imported under a unique module name derived from its path
    relative to ``directory`` and inserted into ``sys.modules``.
    """
    root = Path(directory).resolve()
    if not root.exists():
        raise FileNotFoundError(f"[User Task Loader] Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"[User Task Loader] Path is not a directory: {root}")

    modules: list[ModuleType] = []
    for file_path in sorted(root.glob("**/*.py")):
        module_name = _module_name_for(file_path, root)
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not create a module spec for {file_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            sys.modules.pop(module_name, None)
            raise ImportError(f"Failed to import {file_path}: {e}") from e
        modules.append(module)
    return modules


def register_tasks_from_module(module: ModuleType, registry: Registry | None = None) -> None:
    """Invoke ``register_tasks(registry)`` on ``module`` if it defines it.

    Modules without a ``register_tasks`` entrypoint are ignored.
    """
    registry = registry if registry is not None else global_registry()
    register_fn = getattr(module, REGISTER_TASKS_ENTRYPOINT, None)
    if register_fn is None:
        logger.info(f"[User Task Loader] Skipping {module.__name__} - no {REGISTER_TASKS_ENTRYPOINT}() present.")
        return
    register_fn(registry)
    logger.info(f"[User Task Loader] Registered tasks from {module.__name__}")


def register_tasks_from_modules(modules: Sequence[ModuleType], registry: Registry | None = None) -> None:
    """Register tasks from each module in ``modules`` (see ``register_tasks_from_module``)."""
    for module in modules:
        register_tasks_from_module(module, registry=registry)


def load_extra_tasks(directory: str | os.PathLike, registry: Registry | None = None) -> None:
    """Load and register user-defined tasks from a directory of plugin modules.

    Every ``.py`` file under ``directory`` (searched recursively) is imported as a
    module, and any module exposing a ``register_tasks(registry)`` function has it
    invoked to register its tasks.
    """
    modules = load_modules_from_directory(directory)
    register_tasks_from_modules(modules, registry=registry)
