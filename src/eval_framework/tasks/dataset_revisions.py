"""Pinned Hugging Face dataset revisions.

A lock file maps dataset paths to pinned commit SHAs, so an eval runs against a reproducible
dataset revision. ``BaseTask`` resolves its pin from the lock file named by ``REVISION_LOCKFILE``;
composed benchmarks take a ``Pinned`` policy instead (usually via ``pinned_by_framework``).
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

from huggingface_hub import HfApi

from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy, HfDatasetLoader

logger = logging.getLogger(__name__)

# The revision of the datasets used by the benchmarks is declared in a file, so we can automatically
# update them in CI without having to parse python code.
HF_REVISIONS_LOCKFILE = Path(__file__).resolve().parent / "hf-dataset-revisions.json"

# Hand-maintained pins for datasets that must not move, e.g. because newer revisions are
# incompatible with the task implementation. Never updated by the refresh job.
FROZEN_HF_REVISIONS_LOCKFILE = Path(__file__).resolve().parent / "frozen-hf-dataset-revisions.json"


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

    def revision_for(self, dataset_path: str) -> str:
        """The pinned commit SHA for a dataset. Raises ``KeyError`` if it is not pinned."""
        return self._revisions[dataset_path]

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


@lru_cache
def _revisions_from_file(lockfile: Path) -> HfDatasetRevisions:
    return HfDatasetRevisions.from_file(lockfile)


def pinned_revision(lockfile: Path, dataset_path: str) -> str:
    """The commit SHA pinned for ``dataset_path`` in ``lockfile``.

    Resolves the exact dataset revision an eval runs against, so results stay reproducible
    across dataset updates. Every dataset used by a task is expected to be pinned; a missing
    entry is a bug in the lock file and raises ``KeyError``.
    """
    try:
        return _revisions_from_file(lockfile).revision_for(dataset_path)
    except KeyError:
        raise KeyError(f"Dataset '{dataset_path}' is not pinned in {lockfile}") from None


class Pinned(DatasetPolicy):
    """Pins ``dataset_path`` to the revision recorded for it in ``lockfile``; a run's override still wins."""

    def __init__(self, lockfile: Path, dataset_path: str) -> None:
        self._lockfile = lockfile
        self._dataset_path = dataset_path

    def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
        revision = custom_hf_revision or pinned_revision(self._lockfile, self._dataset_path)
        return HfDatasetLoader(self._dataset_path, revision)

    def documentation(self) -> str:
        url = f"https://huggingface.co/datasets/{self._dataset_path}"
        revision = pinned_revision(self._lockfile, self._dataset_path)
        return f"- Link to dataset: [{url}]({url})\n- Revision: {revision}"


def pinned_by_framework(dataset_path: str) -> Pinned:
    """A ``Pinned`` policy binding ``dataset_path`` to the framework's bundled lock file."""
    return Pinned(HF_REVISIONS_LOCKFILE, dataset_path)
