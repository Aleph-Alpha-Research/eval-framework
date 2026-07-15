"""Tests for dataset revision pinning."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from eval_framework.tasks import dataset_revisions as dr
from tests.tests_eval_framework.tasks.conftest import FIXTURE_REVISIONS


def test_revisions_file_lives_next_to_module() -> None:
    """The lock file path should be the checked-in JSON in the tasks package."""
    module_dir = Path(dr.__file__).resolve().parent
    assert dr.HF_REVISIONS_LOCKFILE.name == "hf-dataset-revisions.json"
    assert dr.HF_REVISIONS_LOCKFILE.parent == module_dir


def test_get_pinned_dataset_revision_returns_sha_for_known_task(fixture_revisions_file: Path) -> None:
    dr.DatasetRevision.reset()
    dr.DatasetRevision.add_revision_file(fixture_revisions_file)

    assert dr.DatasetRevision.pinned_revision("COPA") == FIXTURE_REVISIONS["COPA"]


def test_get_pinned_dataset_revision_returns_none_for_unknown_task(fixture_revisions_file: Path) -> None:
    dr.DatasetRevision.reset()
    dr.DatasetRevision.add_revision_file(fixture_revisions_file)

    assert dr.DatasetRevision.pinned_revision("NotARegisteredTask") is None


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


