"""Tests for dataset revision pinning."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eval_framework.tasks import dataset_revisions as dr


def test_revisions_file_lives_next_to_module() -> None:
    """The lock file path should be the checked-in JSON in the tasks package."""
    module_dir = Path(dr.__file__).resolve().parent
    assert dr.HF_REVISIONS_LOCKFILE.name == "hf-dataset-revisions.json"
    assert dr.HF_REVISIONS_LOCKFILE.parent == module_dir


def test_pinned_revision_returns_sha_for_pinned_dataset(tmp_path: Path) -> None:
    # Given a lock file pinning a dataset
    lockfile = tmp_path / "hf-dataset-revisions.json"
    dr.HfDatasetRevisions({"org/data": "pinned-sha"}).to_file(lockfile)

    # When resolving the dataset's pin, then the pinned SHA is returned
    assert dr.pinned_revision(lockfile, "org/data") == "pinned-sha"


def test_pinned_revision_raises_for_unpinned_dataset(tmp_path: Path) -> None:
    # Given a lock file without a pin for the dataset
    lockfile = tmp_path / "hf-dataset-revisions.json"
    dr.HfDatasetRevisions({}).to_file(lockfile)

    # Then resolving the dataset's pin fails, naming the lock file
    with pytest.raises(KeyError, match="not pinned in .*hf-dataset-revisions.json"):
        dr.pinned_revision(lockfile, "org/data")


def test_update_to_latest_updates_sha() -> None:
    """Each pinned dataset is refreshed to the latest commit SHA."""
    api = MagicMock()
    api.dataset_info.return_value = SimpleNamespace(sha="new-sha")
    revisions = dr.HfDatasetRevisions({"Some/evaltask": "old-sha"})

    revisions.update_to_latest(api)

    assert revisions.to_dict() == {"Some/evaltask": "new-sha"}
    api.dataset_info.assert_called_once_with("Some/evaltask", timeout=100.0)


def test_update_to_latest_keeps_pin_when_lookup_fails() -> None:
    """A failing lookup must not drop the dataset or change its pin."""
    api = MagicMock()
    api.dataset_info.side_effect = RuntimeError("hf outage")
    revisions = dr.HfDatasetRevisions({"Some/evaltask": "old-sha"})

    revisions.update_to_latest(api)

    assert revisions.to_dict() == {"Some/evaltask": "old-sha"}


def test_hf_dataset_revisions_file_round_trip(tmp_path: Path) -> None:
    """Revisions written with ``to_file`` load back identically with ``from_file``."""
    lockfile = tmp_path / "hf-dataset-revisions.json"
    pinned = {"org/b": "sha-b", "org/a": "sha-a"}

    dr.HfDatasetRevisions(pinned).to_file(lockfile)
    loaded = dr.HfDatasetRevisions.from_file(lockfile)

    assert loaded.to_dict() == pinned


def test_hf_dataset_revisions_file_is_sorted_by_dataset_path(tmp_path: Path) -> None:
    """The lock file is written sorted, so refreshing produces stable diffs."""
    lockfile = tmp_path / "hf-dataset-revisions.json"

    dr.HfDatasetRevisions({"org/b": "sha-b", "org/a": "sha-a"}).to_file(lockfile)

    assert lockfile.read_text(encoding="utf-8").index('"org/a"') < lockfile.read_text(encoding="utf-8").index('"org/b"')
