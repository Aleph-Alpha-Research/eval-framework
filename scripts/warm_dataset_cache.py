#!/usr/bin/env python
"""Warm the local dataset cache for CI.

Downloads the raw snapshot of every pinned dataset (both lock files) into the Hugging Face
hub cache. Already-cached revisions are cheap no-ops, so the script serves both full builds
and incremental top-ups after a pin change.

Usage::

    uv run python scripts/warm_dataset_cache.py
"""

import logging

from huggingface_hub import HfApi

from eval_framework.tasks.dataset_revisions import (
    FROZEN_HF_REVISIONS_LOCKFILE,
    HF_REVISIONS_LOCKFILE,
    HfDatasetRevisions,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    api = HfApi()
    for lockfile in (HF_REVISIONS_LOCKFILE, FROZEN_HF_REVISIONS_LOCKFILE):
        HfDatasetRevisions.from_file(lockfile).download_datasets(api)
