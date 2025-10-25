import copy
import tempfile
import traceback
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast
from urllib.parse import urlparse

import wandb
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
        self.notes = ""
        self.settings = kwargs.get("settings") or Settings(mode=kwargs.get("mode", "online"))
        self._logged_artifacts: list[MockArtifact] = []

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
        assert isinstance(artifact_or_path, MockArtifact)
        logged_artifact = copy.deepcopy(artifact_or_path)
        logged_artifact.aliases = aliases or []  # Don't bother here with auto-adding "latest", for test purposes
        self._logged_artifacts.append(logged_artifact)
        return logged_artifact

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
        self._path = Path(urlparse(self.path_uri).path)

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def size(self) -> int:
        return self._path.stat().st_size if self._path.exists() else -1


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
        self._name = name
        self.type = type
        self.description = description
        self.metadata = metadata
        self.incremental = incremental
        self.use_as = use_as
        self._files: list = []
        self.aliases: list[str] = []

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

    @property
    def name(self) -> str:
        return f"{self._name}:{self.aliases[0]}" if self.aliases else self._name

    @property
    def qualified_name(self) -> str:
        return f"test-entity/project/{self.name}"


class MockWandbApi:
    def __init__(self) -> None:
        self.default_entity = "test-entity"

    def artifact(self, name: str) -> MockArtifact | None:
        if wandb.run:
            basename, version = name.split(":", 1)
            for artifact in cast(MockWandbRun, wandb.run)._logged_artifacts:
                if artifact.qualified_name.split(":", 1)[0] == basename and version in artifact.aliases:
                    return artifact
        return None

    def artifact_exists(self, name: str, type: str | None = None) -> bool:
        artifact = self.artifact(name)
        return artifact is not None and (type is None or artifact.type == type)
