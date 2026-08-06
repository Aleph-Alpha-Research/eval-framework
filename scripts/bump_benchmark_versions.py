#!/usr/bin/env python
"""Reconcile ``benchmark-versions.json`` with the current pinned HF revisions.

For each registered task with a ``REVISION_LOCKFILE``, compare its currently
pinned dataset revision against the ``hf_revision`` recorded in the
sibling ``benchmark-versions.json``. When they differ, increment the version
and record the new revision. Tasks never seen before are seeded at version 1.

Usage::

    uv run python scripts/bump_benchmark_versions.py --check
    uv run python scripts/bump_benchmark_versions.py --fix
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import eval_framework.tasks  # noqa: F401  # triggers task registration
from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.benchmark_versions import (
    DEFAULT_VERSION,
    BenchmarkVersions,
    versions_file_for,
)
from eval_framework.tasks.dataset_revisions import pinned_revision
from eval_framework.tasks.registry import registry

logger = logging.getLogger(__name__)


def _collect_tasks_by_versions_file() -> dict[Path, list[type[BaseTask]]]:
    """Group tasks by the versions file they share.

    Multiple lockfiles (e.g. the auto-refreshed one and the frozen one) can live
    in the same directory and therefore share a ``benchmark-versions.json``. We
    still resolve each task's pin against its own ``REVISION_LOCKFILE``.
    """
    grouped: dict[Path, list[type[BaseTask]]] = defaultdict(list)
    for _, factory in registry().items():
        task_cls = factory.task_class()
        if task_cls.REVISION_LOCKFILE is None:
            continue
        grouped[versions_file_for(task_cls.REVISION_LOCKFILE)].append(task_cls)
    return grouped


def _reconcile(versions_file: Path, task_classes: list[type[BaseTask]]) -> tuple[BenchmarkVersions, list[str]]:
    versions = BenchmarkVersions.from_file(versions_file)
    changes: list[str] = []
    live_names: set[str] = set()

    for task_cls in task_classes:
        name = task_cls.NAME
        live_names.add(name)
        lockfile = task_cls.REVISION_LOCKFILE
        assert lockfile is not None  # filtered in _collect_tasks_by_versions_file
        try:
            current_revision = pinned_revision(lockfile, task_cls.DATASET_PATH)
        except KeyError:
            logger.warning("no pin for %s (%s); skipping", name, task_cls.DATASET_PATH)
            continue
        recorded = versions.recorded_revision(name)
        if recorded is None:
            versions.record(name, version=DEFAULT_VERSION, hf_revision=current_revision)
            changes.append(f"seed {name} at v{DEFAULT_VERSION} ({current_revision[:12]})")
        elif recorded != current_revision:
            new_version = versions.version_of(name) + 1
            versions.record(name, version=new_version, hf_revision=current_revision)
            changes.append(
                f"bump {name} v{new_version - 1} -> v{new_version} ({recorded[:12]} -> {current_revision[:12]})"
            )

    for orphan in sorted(set(versions.task_names()) - live_names):
        versions.drop(orphan)
        changes.append(f"drop orphan {orphan}")

    return versions, changes


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail if any entry would change")
    mode.add_argument("--fix", action="store_true", help="write updated entries to disk")
    args = parser.parse_args()

    grouped = _collect_tasks_by_versions_file()
    if not grouped:
        logger.info("no tasks with a REVISION_LOCKFILE; nothing to do")
        return 0

    any_changes = False
    for versions_file, task_classes in sorted(grouped.items()):
        versions, changes = _reconcile(versions_file, task_classes)
        if changes:
            any_changes = True
            logger.info("in %s:", versions_file)
            for change in changes:
                logger.info("  %s", change)
        if args.fix and changes:
            versions.to_file(versions_file)

    if args.check and any_changes:
        logger.error("benchmark-versions.json is out of sync; run --fix")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
