"""Fetch latest Hugging Face dataset commit SHAs for registered tasks.

Overwrites ``task-dataset-revisions.json`` in this package with task class name → SHA.

Usage::

    uv run python -m eval_framework.tasks.benchmarks.dataset_revisions
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

DEFAULT_REVISIONS_FILE = Path(__file__).resolve().parent / "task-dataset-revisions.json"
REVISIONS_FILE = DEFAULT_REVISIONS_FILE


@lru_cache
def _pinned_revisions(revisions_file: Path) -> dict[str, str]:
    return json.loads(revisions_file.read_text(encoding="utf-8"))


def get_pinned_dataset_revision(
    task_class_name: str,
    *,
    revisions_file: Path | None = None,
) -> str | None:
    return _pinned_revisions(revisions_file or REVISIONS_FILE).get(task_class_name)


def _repo_sha(api: HfApi, repo_id: str, cache: dict[str, str | None]) -> str | None:
    if repo_id in cache:
        return cache[repo_id]
    try:
        cache[repo_id] = api.dataset_info(repo_id, timeout=100.0).sha
        logger.info("%s -> %s", repo_id, cache[repo_id])
    except Exception as exc:
        logger.warning("Skipping %s: %s", repo_id, exc)
        cache[repo_id] = None
    return cache[repo_id]


def collect_dataset_revisions(
    task_names: list[str],
    api: HfApi,
) -> dict[str, str]:
    """Return task class name → latest dataset commit SHA for tasks with a Hugging Face path."""
    from eval_framework.tasks.registry import get_task

    cache: dict[str, str | None] = {}
    revisions: dict[str, str] = {}
    for name in task_names:
        try:
            cls = get_task(name)
        except Exception as exc:
            logger.warning("Skipping task %s: %s", name, exc)
            continue
        path = (getattr(cls, "DATASET_PATH", None) or "").strip()
        if path and (sha := _repo_sha(api, path, cache)):
            revisions[cls.__name__] = sha
    return revisions


def main() -> None:
    from eval_framework.tasks.registry import registered_task_names

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    revisions = collect_dataset_revisions(registered_task_names(), HfApi())
    REVISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVISIONS_FILE.write_text(
        json.dumps(dict(sorted(revisions.items())), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %d revisions to %s", len(revisions), REVISIONS_FILE)


if __name__ == "__main__":
    main()
