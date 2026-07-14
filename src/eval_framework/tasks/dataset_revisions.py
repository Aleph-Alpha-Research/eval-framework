"""Refresh the pinned Hugging Face dataset revisions in ``hf-dataset-revisions.json``.

The lock file's keys (dataset paths) define which datasets are pinned; refreshing updates
each pin to the latest commit SHA.

Usage::

    uv run python -m eval_framework.tasks.dataset_revisions
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import HfApi

if TYPE_CHECKING:
    from eval_framework.tasks.registry import EvalFactory

logger = logging.getLogger(__name__)

DEFAULT_REVISIONS_FILE = Path(__file__).resolve().parent / "task-dataset-revisions.json"
REVISIONS_FILE = DEFAULT_REVISIONS_FILE

# The revision of the datasets used by the benchmarks is declared in a file, so we can automatically
# update them in CI without having to parse python code.
HF_REVISIONS_LOCKFILE = Path(__file__).resolve().parent / "hf-dataset-revisions.json"


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
        if cls._INSTANCE is None:
            raise RuntimeError("No revision file added; call add_revision_file() before pinned_revision().")
        return cls._INSTANCE._cache.get(task_class_name)

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


def dataset_revision_collection(
    eval_factories: list["EvalFactory"],
    api: HfApi,
) -> dict[str, str]:
    """Return task class name → latest dataset commit SHA for tasks with a Hugging Face path."""

    cache: dict[str, str | None] = {}
    revisions: dict[str, str] = {}
    for factory in eval_factories:
        path = (factory.dataset_path() or "").strip()
        if path and (sha := _repo_sha(api, path, cache)):
            revisions[factory.task_class().__name__] = sha
    return revisions


class HfDatasetRevisions:
    """Pinned revisions of Hugging Face datasets, mapping dataset path → commit SHA."""

    def __init__(self, revisions: dict[str, str]) -> None:
        self._revisions = dict(revisions)

    @classmethod
    def from_file(cls, path: Path) -> "HfDatasetRevisions":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, str]:
        return dict(self._revisions)

    def to_file(self, path: Path) -> None:
        path.write_text(
            json.dumps(dict(sorted(self._revisions.items())), indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def num_revisions(self) -> int:
        return len(self._revisions)

    def update_to_latest(self, api: HfApi) -> None:
        """Update every pin to its dataset's latest commit SHA.

        Intended to run on CI to ensure the pinned revisions are up-to-date. If the lookup for a
        dataset fails, its existing pin is kept.
        """
        for path, sha in self._revisions.items():
            try:
                latest = api.dataset_info(path, timeout=100.0).sha
            except Exception as exc:
                logger.warning("Could not refresh %s (%s); keeping pinned revision %s", path, exc, sha)
                continue
            if latest and latest != sha:
                logger.info("%s: %s -> %s", path, sha, latest)
            self._revisions[path] = latest or sha


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    revisions = HfDatasetRevisions.from_file(HF_REVISIONS_LOCKFILE)
    revisions.update_to_latest(HfApi())
    revisions.to_file(HF_REVISIONS_LOCKFILE)
    logger.info("Refreshed %d pinned revisions in %s", revisions.num_revisions(), HF_REVISIONS_LOCKFILE)


if __name__ == "__main__":
    main()
