import pytest

def pytest_collection_modifyitems(items):
    """Mark WMT tests to run serially to avoid file I/O race conditions."""
    for item in items:
        if "WMT" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group("wmt_serial"))
