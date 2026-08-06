"""Per-benchmark version integers, bumped by hand when a benchmark's behaviour changes.

Each task's version lives in ``benchmark-versions.json`` next to the
``REVISION_LOCKFILE`` it resolves against. Bumps are manual: when a pinned HF
dataset revision is updated for a task in ``hf-dataset-revisions.json``,
increment its ``version`` in the same PR so downstream consumers can tell
the two runs apart. Entries are ``{task_name: {"version": int}}``; the dict
wrapper leaves room to record further version dimensions later without
another schema churn.
"""

import json
from functools import lru_cache
from pathlib import Path

VERSIONS_FILENAME = "benchmark-versions.json"

# Version returned for tasks with no recorded entry (or no lockfile).
DEFAULT_VERSION = 1


class BenchmarkVersions:
    """Recorded ``{task_name: {"version": int}}`` entries."""

    def __init__(self, entries: dict[str, dict[str, object]]) -> None:
        self._entries = dict(entries)

    @classmethod
    def from_file(cls, path: Path) -> "BenchmarkVersions":
        if not path.exists():
            return cls({})
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def to_file(self, path: Path) -> None:
        payload = json.dumps(self._entries, indent=4, ensure_ascii=False, sort_keys=True) + "\n"
        path.write_text(payload, encoding="utf-8")

    def version_of(self, task_name: str) -> int:
        entry = self._entries.get(task_name)
        if entry is None:
            return DEFAULT_VERSION
        return int(entry["version"])


@lru_cache
def _load(versions_file: Path) -> BenchmarkVersions:
    return BenchmarkVersions.from_file(versions_file)


def versions_file_for(revision_lockfile: Path) -> Path:
    return revision_lockfile.parent / VERSIONS_FILENAME


def version_for(task_name: str, revision_lockfile: Path | None) -> int:
    """The recorded version for ``task_name``, or the default if unrecorded."""
    if revision_lockfile is None:
        return DEFAULT_VERSION
    return _load(versions_file_for(revision_lockfile)).version_of(task_name)
