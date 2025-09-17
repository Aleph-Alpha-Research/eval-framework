from collections.abc import Generator

import pytest

import wandb
from tests.mock_wandb import MockWandbRun


@pytest.fixture
def wandb_run() -> Generator[wandb.Run, None, None]:
    with wandb.init(project="test-project") as run:
        yield run


class TestMockWandbRun:
    def test_init(self, wandb_run: MockWandbRun) -> None:
        assert wandb_run.project == "test-project"
        assert wandb_run.config == {}
        assert wandb_run.summary == {}
        assert wandb_run.name == "mock_run"
        assert wandb_run.logged_data == []
        assert not wandb_run._finished

    def test_log(self, wandb_run: MockWandbRun) -> None:
        test_data = {"accuracy": 0.95, "loss": 0.05}
        wandb_run.log(test_data)

        assert len(wandb_run.logged_data) == 1
        assert wandb_run.logged_data[0]["data"] == test_data
        assert wandb_run.logged_data[0]["step"] is None
        assert wandb_run.logged_data[0]["commit"] is True

    def test_log_with_step(self, wandb_run: MockWandbRun) -> None:
        test_data = {"accuracy": 0.95}
        wandb_run.log(test_data, step=100)
        assert wandb_run.logged_data[0]["step"] == 100

    def test_multiple_logs(self, wandb_run: MockWandbRun) -> None:
        wandb_run.log({"metric1": 1.0})
        wandb_run.log({"metric2": 2.0})

        assert len(wandb_run.logged_data) == 2
        assert wandb_run.logged_data[0]["data"] == {"metric1": 1.0}
        assert wandb_run.logged_data[1]["data"] == {"metric2": 2.0}

    def test_finish(self, wandb_run: MockWandbRun) -> None:
        wandb_run.log({"metric": 1.0})

        wandb_run.finish()

        assert wandb_run._finished

        # Should not log after finishing
        wandb_run.log({"metric": 2.0})
        assert len(wandb_run.logged_data) == 1  # Still only one entry

    def test_get_logged_data(self, wandb_run: MockWandbRun) -> None:
        test_data = {"accuracy": 0.95}

        wandb_run.log(test_data)
        logged_data = wandb_run.get_logged_data()

        assert logged_data == wandb_run.logged_data
        assert logged_data[0]["data"] == test_data
