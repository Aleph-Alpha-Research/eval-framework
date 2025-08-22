import traceback
from types import TracebackType
from typing import Any, Dict, Literal, Optional, Sequence, TypedDict, Unpack

from wandb import Settings
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
    filters: Optional[Dict[str, Any]]
    order: Optional[str]
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

    def finish(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self._finished = True

    def get_logged_data(self) -> list[dict]:
        return self.logged_data


class MockWandb:
    def __init__(self) -> None:
        self.run: MockWandbRun | None = None
        self._login_called = False

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
