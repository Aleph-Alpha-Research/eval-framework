class MockWandbRun:
    def __init__(self, project=None, **kwargs):
        self.config = {}
        self.summary = {}
        self.name = "mock_run"
        self.project = project
        self.logged_data = []  # Store all logged data for testing
        self._finished = False

    def log(self, data, step=None, commit=True):
        """Mock wandb.log() - stores data for verification in tests"""
        if not self._finished:
            log_entry = {"data": data, "step": step, "commit": commit}
            self.logged_data.append(log_entry)

    def finish(self):
        """Mock wandb.finish()"""
        self._finished = True

    def get_logged_data(self):
        """Helper method for tests to verify logged data"""
        return self.logged_data


class MockWandb:
    def __init__(self):
        self.run = None
        self._login_called = False

    def init(self, project=None, **kwargs):
        """Mock wandb.init() - returns a mock run object"""
        self.run = MockWandbRun(project=project, **kwargs)
        return self.run

    def log(self, data, step=None, commit=True):
        """Mock wandb.log() - delegates to current run if available"""
        if self.run:
            self.run.log(data, step, commit)

    def login(self, key=None, **kwargs):
        """Mock wandb.login()"""
        self._login_called = True

    def finish(self):
        """Mock wandb.finish()"""
        if self.run:
            self.run.finish()
