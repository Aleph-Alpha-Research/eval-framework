from tests.mock_wandb import MockWandb, MockWandbRun


class TestMockWandbRun:
    def test_init(self) -> None:
        run = MockWandbRun(project="test-project")
        assert run.project == "test-project"
        assert run.config == {}
        assert run.summary == {}
        assert run.name == "mock_run"
        assert run.logged_data == []
        assert not run._finished

    def test_log(self) -> None:
        run = MockWandbRun()
        test_data = {"accuracy": 0.95, "loss": 0.05}

        run.log(test_data)

        assert len(run.logged_data) == 1
        assert run.logged_data[0]["data"] == test_data
        assert run.logged_data[0]["step"] is None
        assert run.logged_data[0]["commit"] is True

    def test_log_with_step(self) -> None:
        run = MockWandbRun()
        test_data = {"accuracy": 0.95}

        run.log(test_data, step=100)

        assert run.logged_data[0]["step"] == 100

    def test_multiple_logs(self) -> None:
        run = MockWandbRun()

        run.log({"metric1": 1.0})
        run.log({"metric2": 2.0})

        assert len(run.logged_data) == 2
        assert run.logged_data[0]["data"] == {"metric1": 1.0}
        assert run.logged_data[1]["data"] == {"metric2": 2.0}

    def test_finish(self) -> None:
        run = MockWandbRun()
        run.log({"metric": 1.0})

        run.finish()

        assert run._finished

        # Should not log after finishing
        run.log({"metric": 2.0})
        assert len(run.logged_data) == 1  # Still only one entry

    def test_get_logged_data(self) -> None:
        run = MockWandbRun()
        test_data = {"accuracy": 0.95}

        run.log(test_data)
        logged_data = run.get_logged_data()

        assert logged_data == run.logged_data
        assert logged_data[0]["data"] == test_data


class TestMockWandb:
    def test_init(self) -> None:
        wandb_mock = MockWandb()
        assert wandb_mock.run is None
        assert not wandb_mock._login_called

    def test_init_run(self) -> None:
        wandb_mock = MockWandb()

        run = wandb_mock.init(project="test-project")

        assert wandb_mock.run is not None
        assert run.project == "test-project"
        assert isinstance(run, MockWandbRun)

    def test_login(self) -> None:
        wandb_mock = MockWandb()

        wandb_mock.login(key="test-key")

        assert wandb_mock._login_called

    def test_log_without_run(self) -> None:
        wandb_mock = MockWandb()

        # Should not raise an error
        wandb_mock.log({"metric": 1.0})

    def test_log_with_run(self) -> None:
        wandb_mock = MockWandb()
        wandb_mock.init(project="test")

        wandb_mock.log({"metric": 1.0})

        assert wandb_mock.run is not None
        assert len(wandb_mock.run.logged_data) == 1
        assert wandb_mock.run.logged_data[0]["data"] == {"metric": 1.0}

    def test_finish_without_run(self) -> None:
        wandb_mock = MockWandb()
        wandb_mock.finish()

    def test_finish_with_run(self) -> None:
        wandb_mock = MockWandb()
        wandb_mock.init(project="test")

        wandb_mock.finish()

        assert wandb_mock.run is not None
        assert wandb_mock.run._finished
