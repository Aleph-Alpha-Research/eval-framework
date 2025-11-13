#!/usr/bin/env python
"""Dataset cache management utilities.

Commands:
    update  - Download only changed HF datasets (incremental), always ensures sacrebleu is cached
    rebuild - Download ALL HF datasets (full rebuild), always ensures sacrebleu is cached

Note: Sacrebleu WMT datasets are automatically cached as part of both commands.
      They use a separate cache (SACREBLEU env var) but are stored alongside HF datasets.
"""

import argparse

from eval_framework.tasks.task_names import (
    make_sure_all_hf_datasets_are_in_cache,
    save_hf_dataset_commits,
    update_changed_datasets_only,
)


def update_datasets() -> None:
    """Incremental update: download only changed datasets."""
    updates_made, _ = update_changed_datasets_only(verbose=True)
    if updates_made:
        save_hf_dataset_commits()


def rebuild_all() -> None:
    """Full rebuild: download all datasets from scratch."""
    make_sure_all_hf_datasets_are_in_cache()
    save_hf_dataset_commits()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset cache management utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Both commands ensure sacrebleu WMT datasets are cached alongside HF datasets.",
    )
    parser.add_argument(
        "command",
        choices=["update", "rebuild"],
        help="'update' for incremental updates, 'rebuild' for full cache rebuild",
    )
    args = parser.parse_args()

    if args.command == "update":
        update_datasets()
    elif args.command == "rebuild":
        rebuild_all()
