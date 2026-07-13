"""Tests for dataset revision pinning."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from eval_framework.metrics.base import BaseMetric
from eval_framework.tasks import dataset_revisions as dr
from eval_framework.tasks.base import BaseTask, ResponseType
from eval_framework.tasks.perturbation import PerturbationConfig
from eval_framework.tasks.registry import EvalFactory
from tests.tests_eval_framework.tasks.conftest import FIXTURE_REVISIONS


class EvalFactoryDouble(EvalFactory):
    def task_class(self) -> type[BaseTask]:
        raise NotImplementedError

    @property
    def source_module(self) -> str:
        raise NotImplementedError

    def response_type(self) -> ResponseType:
        raise NotImplementedError

    def metrics(self) -> list[type["BaseMetric"]]:
        raise NotImplementedError

    def display_name(self) -> str:
        raise NotImplementedError

    def dataset_path(self) -> str | None:
        raise NotImplementedError

    def create(self, num_fewshot: int, custom_subjects: list[str] | None, custom_hf_revision: str | None) -> BaseTask:
        raise NotImplementedError

    def create_perturbation(
        self,
        perturbation_config: PerturbationConfig,
        num_fewshot: int,
        custom_subjects: list[str] | None,
        custom_hf_revision: str | None,
    ) -> BaseTask:
        raise NotImplementedError


def test_revisions_file_lives_next_to_module() -> None:
    """The default output path should be the checked-in JSON in tasks/benchmarks."""
    module_dir = Path(dr.__file__).resolve().parent
    assert dr.DEFAULT_REVISIONS_FILE.name == "task-dataset-revisions.json"
    assert dr.DEFAULT_REVISIONS_FILE.parent == module_dir


def test_dataset_revision_collection_fetches_sha_for_hf_task() -> None:
    """A task with a DATASET_PATH should appear in the result keyed by class name."""

    class SomeEvalTask(BaseTask): ...

    class SomeEvalTaskFactory(EvalFactoryDouble):
        def task_class(self) -> type[BaseTask]:
            return SomeEvalTask

        def dataset_path(self) -> str | None:
            return "Some/evaltask"

    api = MagicMock()
    api.dataset_info.return_value = SimpleNamespace(sha="abc123")

    revisions = dr.dataset_revision_collection([SomeEvalTaskFactory()], api)

    assert revisions == {"SomeEvalTask": "abc123"}
    api.dataset_info.assert_called_once_with("Some/evaltask", timeout=100.0)


def test_dataset_revision_collection_skips_task_without_dataset_path() -> None:
    """Tasks with an empty DATASET_PATH are omitted from the output."""

    class NoDataSetEvalTask(BaseTask): ...

    class NoDataSetEvalTaskFactory(EvalFactoryDouble):
        def task_class(self) -> type[BaseTask]:
            return NoDataSetEvalTask

        def dataset_path(self) -> str | None:
            return ""

    api = MagicMock()

    revisions = dr.dataset_revision_collection([NoDataSetEvalTaskFactory()], api)

    assert revisions == {}
    api.dataset_info.assert_not_called()


def test_dataset_revision_collection_skips_failed_dataset_lookup() -> None:
    """If the Hugging Face API fails for a dataset, that task is omitted."""

    class SomeEvalTask(BaseTask): ...

    class SomeEvalTaskFactory(EvalFactoryDouble):
        def task_class(self) -> type[BaseTask]:
            return SomeEvalTask

        def dataset_path(self) -> str | None:
            return "Some/evaltask"

    api = MagicMock()
    api.dataset_info.side_effect = RuntimeError("not found")

    revisions = dr.dataset_revision_collection([SomeEvalTaskFactory()], api)

    assert revisions == {}


def test_get_pinned_dataset_revision_returns_sha_for_known_task(fixture_revisions_file: Path) -> None:
    dr.DatasetRevision.reset()
    dr.DatasetRevision.add_revision_file(fixture_revisions_file)

    assert dr.DatasetRevision.pinned_revision("COPA") == FIXTURE_REVISIONS["COPA"]


def test_get_pinned_dataset_revision_returns_none_for_unknown_task(fixture_revisions_file: Path) -> None:
    dr.DatasetRevision.reset()
    dr.DatasetRevision.add_revision_file(fixture_revisions_file)

    assert dr.DatasetRevision.pinned_revision("NotARegisteredTask") is None


def test_dataset_revision_collection_reuses_sha_for_shared_dataset() -> None:
    """Multiple tasks sharing one DATASET_PATH should trigger a single API call."""

    class SomeEvalTask(BaseTask): ...

    class SomeEvalTaskFactory(EvalFactoryDouble):
        def task_class(self) -> type[BaseTask]:
            return SomeEvalTask

        def dataset_path(self) -> str | None:
            return "Some/evaltask"

    # Based on same dataset but different task styler
    class SomeEvalTaskMC(BaseTask): ...

    class SomeEvalTaskMCFactory(EvalFactoryDouble):
        def task_class(self) -> type[BaseTask]:
            return SomeEvalTaskMC

        def dataset_path(self) -> str | None:
            return "Some/evaltask"

    api = MagicMock()
    api.dataset_info.return_value = SimpleNamespace(sha="shared-sha")

    revisions = dr.dataset_revision_collection([SomeEvalTaskFactory(), SomeEvalTaskMCFactory()], api)

    assert revisions == {"SomeEvalTask": "shared-sha", "SomeEvalTaskMC": "shared-sha"}
    api.dataset_info.assert_called_once()
