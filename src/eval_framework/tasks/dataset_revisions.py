"""Fetch latest Hugging Face dataset commit SHAs for registered tasks.

Overwrites ``task-dataset-revisions.json`` in this package with task class name → SHA.

Usage::

    uv run python -m eval_framework.tasks.dataset_revisions
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

DEFAULT_REVISIONS_FILE = Path(__file__).resolve().parent / "task-dataset-revisions.json"
REVISIONS_FILE = DEFAULT_REVISIONS_FILE


@lru_cache
def _pinned_revisions(revisions_file: Path) -> dict[str, str]:
    return json.loads(revisions_file.read_text(encoding="utf-8"))


class DatasetRevision:
    _INSTANCE: "DatasetRevision | None" = None

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    @classmethod
    def _get_instance(cls) -> "DatasetRevision":
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE

    @classmethod
    def add_revision_file(cls, file_path: Path | str) -> None:
        instance = cls._get_instance()
        instance._append_revision_file(Path(file_path))

    @classmethod
    def pinned_revision(cls, task_class_name: str) -> str | None:
        return cls._get_instance()._cache.get(task_class_name)

    @classmethod
    def reset(cls) -> None:
        # for unit tests only.
        cls._INSTANCE = None

    def _append_revision_file(self, file_path: Path) -> None:
        revisions = _pinned_revisions(file_path)
        self._cache |= revisions


def _repo_sha(api: HfApi, repo_id: str, cache: dict[str, str | None]) -> str | None:
    if repo_id in cache:
        return cache[repo_id]
    try:
        cache[repo_id] = api.dataset_info(repo_id, timeout=100.0).sha
        logger.info("%s -> %s", repo_id, cache[repo_id])
    except Exception as exc:
        logger.warning("Skipping %s: %s", repo_id, exc)
        cache[repo_id] = None
    return cache[repo_id]


def collect_dataset_revisions(
    task_names: list[str],
    api: HfApi,
) -> dict[str, str]:
    """Return task class name → latest dataset commit SHA for tasks with a Hugging Face path."""
    from eval_framework.tasks.registry import get_task

    cache: dict[str, str | None] = {}
    revisions: dict[str, str] = {}
    for name in task_names:
        try:
            cls = get_task(name)
        except Exception as exc:
            logger.warning("Skipping task %s: %s", name, exc)
            continue
        path = (getattr(cls, "DATASET_PATH", None) or "").strip()
        if path and (sha := _repo_sha(api, path, cache)):
            revisions[cls.__name__] = sha
    return revisions


def main() -> None:
    from eval_framework.tasks.registry import registered_task_names

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    revisions = collect_dataset_revisions(registered_task_names(), HfApi())
    REVISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVISIONS_FILE.write_text(
        json.dumps(dict(sorted(revisions.items())), indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %d revisions to %s", len(revisions), REVISIONS_FILE)


if __name__ == "__main__":
    main()
