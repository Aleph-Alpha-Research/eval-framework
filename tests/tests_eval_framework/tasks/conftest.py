import json
from pathlib import Path

import pytest

FIXTURE_REVISIONS: dict[str, str] = {
    "COPA_SuperGLUE_EN_Cloze": "abcdef0123456789abcdef0123456789abcdef01",
}


def write_fixture_revisions_file(directory: Path) -> Path:
    path = directory / "task-dataset-revisions.json"
    path.write_text(json.dumps(FIXTURE_REVISIONS, indent=4) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def fixture_revisions_file(tmp_path: Path) -> Path:
    return write_fixture_revisions_file(tmp_path)
