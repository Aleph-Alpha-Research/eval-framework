"""Tests for dataset revision pinning script."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from eval_framework.tasks.benchmarks import update_dataset_revisions as udr


def test_revisions_file_lives_next_to_update_script() -> None:
    """The default output path should be the checked-in JSON in tasks/benchmarks."""
    # Given
    expected_name = "task-dataset-revisions.json"
    script_dir = Path(udr.__file__).resolve().parent

    # When
    output_file = udr.REVISIONS_FILE

    # Then
    assert output_file.name == expected_name
    assert output_file.parent == script_dir


def test_collect_dataset_revisions_fetches_sha_for_hf_task() -> None:
    """A task with a DATASET_PATH should appear in the result keyed by class name."""

    # Given
    class CoQA:
        __name__ = "CoQA"
        DATASET_PATH = "EleutherAI/coqa"

    api = MagicMock()
    api.dataset_info.return_value = SimpleNamespace(sha="abc123")

    # When
    revisions = udr.collect_dataset_revisions(["CoQA"], api, get_task_fn=lambda _: CoQA)

    # Then
    assert revisions == {"CoQA": "abc123"}
    api.dataset_info.assert_called_once_with("EleutherAI/coqa", timeout=100.0)


def test_collect_dataset_revisions_skips_task_without_dataset_path() -> None:
    """Tasks with an empty DATASET_PATH are omitted from the output."""

    # Given
    class NoDataset:
        __name__ = "NoDataset"
        DATASET_PATH = ""

    api = MagicMock()

    # When
    revisions = udr.collect_dataset_revisions(["NoDataset"], api, get_task_fn=lambda _: NoDataset)

    # Then
    assert revisions == {}
    api.dataset_info.assert_not_called()


def test_collect_dataset_revisions_skips_failed_task_load() -> None:
    """If loading a task class fails, the script continues without that task."""
    # Given
    api = MagicMock()

    def failing_get_task(_: str) -> type:
        raise ImportError("broken import")

    # When
    revisions = udr.collect_dataset_revisions(["BrokenTask"], api, get_task_fn=failing_get_task)

    # Then
    assert revisions == {}
    api.dataset_info.assert_not_called()


def test_collect_dataset_revisions_skips_failed_dataset_lookup() -> None:
    """If the Hugging Face API fails for a dataset, that task is omitted."""

    # Given
    class CoQA:
        __name__ = "CoQA"
        DATASET_PATH = "EleutherAI/coqa"

    api = MagicMock()
    api.dataset_info.side_effect = RuntimeError("not found")

    # When
    revisions = udr.collect_dataset_revisions(["CoQA"], api, get_task_fn=lambda _: CoQA)

    # Then
    assert revisions == {}


def test_collect_dataset_revisions_reuses_sha_for_shared_dataset() -> None:
    """Multiple tasks sharing one DATASET_PATH should trigger a single API call."""

    # Given
    class CoQA:
        __name__ = "CoQA"
        DATASET_PATH = "EleutherAI/coqa"

    class CoQAMC:
        __name__ = "CoQAMC"
        DATASET_PATH = "EleutherAI/coqa"

    api = MagicMock()
    api.dataset_info.return_value = SimpleNamespace(sha="shared-sha")

    # When
    revisions = udr.collect_dataset_revisions(
        ["CoQA", "CoQAMC"],
        api,
        get_task_fn=lambda name: CoQA if name == "CoQA" else CoQAMC,
    )

    # Then
    assert revisions == {"CoQA": "shared-sha", "CoQAMC": "shared-sha"}
    api.dataset_info.assert_called_once()
