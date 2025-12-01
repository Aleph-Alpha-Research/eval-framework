from typing import Any, Final

import pytest

NO_SKIP_OPTION: Final[str] = "--no-skip"


def pytest_addoption(parser: Any) -> None:
    parser.addoption(NO_SKIP_OPTION, action="store_true", default=False, help="also run skipped tests")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Any:
    outcome = yield
    rep = outcome.get_result()

    # Turn skipped tests into failed tests
    if item.config.getoption(NO_SKIP_OPTION):
        if rep.skipped and call.excinfo.errisinstance(pytest.skip.Exception):
            rep.outcome = "failed"
            r = call.excinfo._getreprcrash()
            rep.longrepr = f"Forbidden skipped test - {r.message}"
