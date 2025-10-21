import tempfile
import traceback
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from wandb import Settings
from wandb.sdk.lib.paths import StrPath


class MockWandbRun:
    def __init__(self, **kwargs: Any):
        self.config: dict = {}
        self.summary: dict = {}
        self.name: str = "mock_run"
        self.project: str = kwargs.get("project", "")
        self.logged_data: list[dict] = []  # Store all logged data for testing
        self._finished: bool = False
        self.id: str = str(kwargs.get("id") or "mock_run_id")
        self._logged_artifacts: list[dict[str, Any]] = []
        self.notes = ""
        self.settings = kwargs.get("settings") or Settings(mode=kwargs.get("mode", "online"))

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
        artifact_or_path: "MockArtifact" | StrPath,
        name: str | None = None,
        type: str | None = None,
        aliases: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> "MockArtifact":
        if isinstance(artifact_or_path, str):
            artifact = MockArtifact(artifact_or_path, "mock_artifact")
        elif isinstance(artifact_or_path, MockArtifact):
            artifact = artifact_or_path
        else:
            # Handle Path-like objects
            artifact = MockArtifact(str(artifact_or_path), "mock_artifact")
        self._logged_artifacts.append(dict(artifact=artifact, aliases=aliases or []))
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
        self.Artifact = MockArtifact  # Make Artifact class available

    def init(self, **kwargs: Any) -> MockWandbRun:
        self.run = MockWandbRun(**kwargs)
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
        if isinstance(artifact, str):
            return MockArtifact(artifact, "mock_artifact")
        return artifact

    def log_artifact(
        self,
        artifact_or_path: "MockArtifact" | StrPath,
        name: str | None = None,
        type: str | None = None,
        aliases: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> "MockArtifact":
        if self.run:
            return self.run.log_artifact(artifact_or_path, name, type, aliases, tags)
        raise RuntimeError("No active run to log artifact to.")


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
        self._files: list = []

    def files(self) -> Sequence[MockArtifactFile]:
        return self._files

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
        self._files.append(MockArtifactFile(uri))
        return [uri]

    def add_file(
        self,
        local_path: str,
        name: str | None = None,
        is_tmp: bool | None = False,
        skip_cache: bool | None = False,
        policy: Literal["mutable", "immutable"] | None = "mutable",
        overwrite: bool = False,
    ) -> None:
        # Mock implementation: just return None instead of `ArtifactManifestEntry`
        self._files.append(MockArtifactFile(Path(local_path).name if name is None else name))


class MockWandbApi:
    def __init__(self) -> None:
        self.default_entity = "test-entity"
        self._artifacts: dict[str, MockArtifact] = {}

    def artifact(self, name: str) -> MockArtifact | None:
        return self.get_artifact(name)

    def set_artifact(self, artifact_id: str, file_list: list[str]) -> None:
        # used only for testing purposes
        artifact = MockArtifact(artifact_id, "model")
        for file in file_list:
            artifact.add_reference(file)
        self._artifacts[f"wandb-registry-model/{artifact_id}:latest"] = artifact

    def get_artifact(self, artifact_id: str) -> MockArtifact | None:
        return self._artifacts.get(artifact_id)
