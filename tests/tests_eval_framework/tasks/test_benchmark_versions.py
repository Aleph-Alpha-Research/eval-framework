"""Tests for per-benchmark version tracking."""

from pathlib import Path

from eval_framework.tasks import benchmark_versions as bv


def test_versions_file_lives_next_to_lockfile() -> None:
    # Given a lockfile path
    lockfile = Path("/tmp/some-dir/hf-dataset-revisions.json")

    # When deriving the versions file path
    versions_file = bv.versions_file_for(lockfile)

    # Then it sits next to the lockfile with the standard filename
    assert versions_file == lockfile.parent / bv.VERSIONS_FILENAME


def test_version_for_returns_default_when_no_lockfile() -> None:
    # Given no lockfile (task without a HF dataset)
    # When looking up its version
    version = bv.version_for("AnyTask", None)

    # Then the default version is returned
    assert version == bv.DEFAULT_VERSION


def test_version_for_returns_default_when_task_absent(tmp_path: Path) -> None:
    # Given a versions file that does not list the task
    bv.BenchmarkVersions({}).to_file(tmp_path / bv.VERSIONS_FILENAME)
    bv._load.cache_clear()

    # When looking up the version, then the default is returned
    assert bv.version_for("AnyTask", tmp_path / "hf-dataset-revisions.json") == bv.DEFAULT_VERSION


def test_version_for_returns_recorded_version(tmp_path: Path) -> None:
    # Given a versions file with a recorded task version
    entries = {"AnyTask": {"version": 3}}
    bv.BenchmarkVersions(entries).to_file(tmp_path / bv.VERSIONS_FILENAME)
    bv._load.cache_clear()

    # When looking up the version, then the recorded integer is returned
    assert bv.version_for("AnyTask", tmp_path / "hf-dataset-revisions.json") == 3
