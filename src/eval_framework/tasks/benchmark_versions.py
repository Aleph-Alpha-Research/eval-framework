"""Per-benchmark version integers, bumped when a benchmark's behaviour changes.

A benchmark's version lives in ``benchmark-versions.json`` next to the
``REVISION_LOCKFILE`` a task resolves against. Today only a pinned HF dataset
revision change triggers a bump; further behaviour dimensions can be added
without touching task classes.
"""

import json
from functools import lru_cache
from pathlib import Path

VERSIONS_FILENAME = "benchmark-versions.json"

# Version returned when a task has no entry (or no lockfile). Every task starts
# here on first observation.
DEFAULT_VERSION = 1


class BenchmarkVersions:
    """Recorded ``{task_name: {"version": int, "hf_revision": str}}`` entries."""

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

    def recorded_revision(self, task_name: str) -> str | None:
        entry = self._entries.get(task_name)
        if entry is None:
            return None
        revision = entry.get("hf_revision")
        return str(revision) if revision is not None else None

    def record(self, task_name: str, *, version: int, hf_revision: str | None) -> None:
        self._entries[task_name] = {"version": version, "hf_revision": hf_revision}

    def drop(self, task_name: str) -> None:
        self._entries.pop(task_name, None)

    def task_names(self) -> list[str]:
        return sorted(self._entries)


@lru_cache
def _load(versions_file: Path) -> BenchmarkVersions:
    return BenchmarkVersions.from_file(versions_file)


def versions_file_for(revision_lockfile: Path) -> Path:
    return revision_lockfile.parent / VERSIONS_FILENAME


def version_for(task_name: str, revision_lockfile: Path | None) -> int:
    """The recorded version for ``task_name``, or the default if unrecorded.

    Tasks without a HF revision lockfile have no behaviour dimension we track
    today, so they always return the default version.
    """
    if revision_lockfile is None:
        return DEFAULT_VERSION
    return _load(versions_file_for(revision_lockfile)).version_of(task_name)
