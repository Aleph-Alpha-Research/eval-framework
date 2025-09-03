import tempfile
import traceback
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypedDict, Unpack

from wandb import Artifact, Settings
from wandb.apis.public import RetryingClient
from wandb.sdk.lib.paths import StrPath


class InitKwargs(TypedDict, total=False):
    entity: str | None
    project: str | None
    dir: StrPath | None
    id: str | None
    name: str | None
    notes: str | None
    tags: Sequence[str] | None
    config: dict[str, Any] | str | None
    config_exclude_keys: list[str] | None
    config_include_keys: list[str] | None
    allow_val_change: bool | None
    group: str | None
    job_type: str | None
    mode: Literal["online", "offline", "disabled"] | None
    force: bool | None
    anonymous: Literal["never", "allow", "must"] | None
    reinit: bool | Literal[None, "default", "return_previous", "finish_previous", "create_new"]
    resume: bool | Literal["allow", "never", "must", "auto"] | None
    resume_from: str | None
    fork_from: str | None
    save_code: bool | None
    tensorboard: bool | None
    sync_tensorboard: bool | None
    monitor_gym: bool | None
    settings: Settings | dict[str, Any] | None


class RunInitKwargs(TypedDict, total=False):
    client: RetryingClient | None
    entity: str | None
    project: str | None
    filters: dict[str, Any] | None
    order: str | None
    per_page: int
    include_sweeps: bool


class MockWandbRun:
    def __init__(self, **kwargs: Unpack[RunInitKwargs]):
        self.config: dict = {}
        self.summary: dict = {}
        self.name: str = "mock_run"
        self.project: str = kwargs.get("project") or ""
        self.logged_data: list[dict] = []  # Store all logged data for testing
        self._finished: bool = False
        self.id: str = str(kwargs.get("id") or "mock_run_id")
        self.logged_artifacts = []

    def __enter__(self) -> "MockWandbRun":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_val: BaseException,
        exc_tb: TracebackType,
    ) -> bool:
        exception_raised = exc_type is not None
        if exception_raised:
            traceback.print_exception(exc_type, exc_val, exc_tb)
        exit_code = 1 if exception_raised else 0
        self.finish(exit_code=exit_code)
        return not exception_raised

    def log(self, data: dict, step: int | None = None, commit: bool = True) -> None:
        if not self._finished:
            log_entry = {"data": data, "step": step, "commit": commit}
            self.logged_data.append(log_entry)

    def log_artifact(
        self,
        artifact_or_path: Artifact | StrPath,
        name: str | None = None,
        type: str | None = None,
        aliases: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> "MockArtifact":
        if isinstance(artifact_or_path, str):
            artifact = MockArtifact(artifact_or_path, "mock_artifact")
        else:
            artifact = artifact_or_path
        self.logged_artifacts.append(artifact)
        return artifact

    def finish(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self._finished = True

    def mark_preempting(self) -> None:
        pass

    def get_logged_data(self) -> list[dict]:
        return self.logged_data


class MockWandb:
    def __init__(self) -> None:
        self.run: MockWandbRun | None = None
        self._login_called = False
        self.Api = MockWandbApi  # Make Api class available

    def init(self, **kwargs: Unpack[InitKwargs]) -> MockWandbRun:
        # get the intersection between initkwargs and runinitkwargs.
        # some are used in the actual init and others are generated. we just need the intersection for tests.
        run_kwargs: dict[str, Any] = {k: v for k, v in kwargs.items() if k in RunInitKwargs.__annotations__}
        self.run = MockWandbRun(**run_kwargs)
        return self.run

    def log(self, data: dict, step: int | None = None, commit: bool = True) -> None:
        if self.run:
            self.run.log(data, step, commit)

    def login(self, key: str | None = None, **kwargs: Any) -> None:
        self._login_called = True

    def finish(self, exit_code: int = 0) -> None:
        if self.run:
            self.run.finish(exit_code)

    def use_artifact(self, artifact: "MockArtifact | str") -> "MockArtifact":
        """Mock wandb.use_artifact function"""
        if isinstance(artifact, str):
            # Create a mock artifact from string
            return MockArtifact(artifact)
        return artifact


class MockArtifactFile:
    def __init__(self, path_uri: str):
        self.path_uri = path_uri

    @property
    def name(self) -> str:
        return Path(self.path_uri).name


class MockArtifact:
    def __init__(
        self,
        name: str,
        type: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        incremental: bool = False,
        use_as: str | None = None,
    ) -> None:
        self.id = name
        self.name = name
        self.type = type
        self.description = description
        self.metadata = metadata
        self.incremental = incremental
        self.use_as = use_as
        self.files_ = []

    def files(self):
        return self.files_

    def download(
        self,
        root: StrPath | None = None,
        allow_missing_references: bool = False,
        skip_cache: bool | None = None,
        path_prefix: StrPath | None = None,
        multipart: bool | None = None,
    ) -> str:
        if root:
            return str(root)
        # Return a temporary directory path for testing
        return tempfile.mkdtemp()

    def add_reference(
        self, uri: str, name: StrPath | None = None, checksum: bool = True, max_objects: int | None = None
    ) -> Sequence[str]:
        # Mock implementation: just return the URI for testing
        self.files_.append(MockArtifactFile(uri))
        return [uri]


class MockWandbApi:
    def __init__(self):
        self.entity = "test-entity"
        self._artifacts = {}

    def artifact(self, name: str) -> MockArtifact:
        return self.get_artifact(name)

    def set_artifact(self, artifact_id: str, file_list: list[str]):
        # used only for testing purposes
        artifact = MockArtifact(artifact_id, "model")
        for file in file_list:
            artifact.add_reference(file)
        self._artifacts[f"wandb-registry-model/{artifact_id}:latest"] = artifact

    def get_artifact(self, artifact_id: str) -> MockArtifact | None:
        return self._artifacts.get(artifact_id)
