"""Tests for dataset revision pinning."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from eval_framework.tasks import dataset_revisions as dr
from eval_framework.tasks.registry import Registry
from tests.tests_eval_framework.tasks.conftest import FIXTURE_REVISIONS


def test_revisions_file_lives_next_to_module() -> None:
    """The default output path should be the checked-in JSON in tasks/benchmarks."""
    module_dir = Path(dr.__file__).resolve().parent
    assert dr.DEFAULT_REVISIONS_FILE.name == "task-dataset-revisions.json"
    assert dr.DEFAULT_REVISIONS_FILE.parent == module_dir


def test_collect_dataset_revisions_fetches_sha_for_hf_task() -> None:
    """A task with a DATASET_PATH should appear in the result keyed by class name."""

    class CoQA:
        NAME = "CoQA"
        DATASET_PATH = "EleutherAI/coqa"

    TEST_REGISTRY = Registry()
    TEST_REGISTRY.add(CoQA)

    api = MagicMock()
    api.dataset_info.return_value = SimpleNamespace(sha="abc123")

    with patch("eval_framework.tasks.registry.registry", return_value=TEST_REGISTRY):
        revisions = dr.collect_dataset_revisions(["CoQA"], api)

    assert revisions == {"CoQA": "abc123"}
    api.dataset_info.assert_called_once_with("EleutherAI/coqa", timeout=100.0)


def test_collect_dataset_revisions_skips_task_without_dataset_path() -> None:
    """Tasks with an empty DATASET_PATH are omitted from the output."""

    class NoDataset:
        NAME = "NoDataset"
        DATASET_PATH = ""

    TEST_REGISTRY = Registry()
    TEST_REGISTRY.add(NoDataset)

    api = MagicMock()

    with patch("eval_framework.tasks.registry.registry", return_value=TEST_REGISTRY):
        revisions = dr.collect_dataset_revisions(["NoDataset"], api)

    assert revisions == {}
    api.dataset_info.assert_not_called()


def test_collect_dataset_revisions_skips_failed_task_load() -> None:
    """If loading a task class fails, the script continues without that task."""
    api = MagicMock()

    with patch("eval_framework.tasks.registry.get_task", side_effect=ImportError("broken import")):
        revisions = dr.collect_dataset_revisions(["BrokenTask"], api)

    assert revisions == {}
    api.dataset_info.assert_not_called()


def test_collect_dataset_revisions_skips_failed_dataset_lookup() -> None:
    """If the Hugging Face API fails for a dataset, that task is omitted."""

    class CoQA:
        __name__ = "CoQA"
        DATASET_PATH = "EleutherAI/coqa"

    api = MagicMock()
    api.dataset_info.side_effect = RuntimeError("not found")

    with patch("eval_framework.tasks.registry.get_task", return_value=CoQA):
        revisions = dr.collect_dataset_revisions(["CoQA"], api)

    assert revisions == {}


def test_get_pinned_dataset_revision_returns_sha_for_known_task(fixture_revisions_file: Path) -> None:
    dr.DatasetRevision.reset()
    dr.DatasetRevision.add_revision_file(fixture_revisions_file)

    assert dr.DatasetRevision.pinned_revision("COPA") == FIXTURE_REVISIONS["COPA"]


def test_get_pinned_dataset_revision_returns_none_for_unknown_task(fixture_revisions_file: Path) -> None:
    dr.DatasetRevision.reset()
    dr.DatasetRevision.add_revision_file(fixture_revisions_file)

    assert dr.DatasetRevision.pinned_revision("NotARegisteredTask") is None


def test_collect_dataset_revisions_reuses_sha_for_shared_dataset() -> None:
    """Multiple tasks sharing one DATASET_PATH should trigger a single API call."""

    class CoQA:
        NAME = "CoQA"
        DATASET_PATH = "EleutherAI/coqa"

    class CoQAMC:
        NAME = "CoQAMC"
        DATASET_PATH = "EleutherAI/coqa"

    TEST_REGISTRY = Registry()
    TEST_REGISTRY.add(CoQA)
    TEST_REGISTRY.add(CoQAMC)

    api = MagicMock()
    api.dataset_info.return_value = SimpleNamespace(sha="shared-sha")

    with patch("eval_framework.tasks.registry.registry", return_value=TEST_REGISTRY):
        revisions = dr.collect_dataset_revisions(["CoQA", "CoQAMC"], api)

    assert revisions == {"CoQA": "shared-sha", "CoQAMC": "shared-sha"}
    api.dataset_info.assert_called_once()
