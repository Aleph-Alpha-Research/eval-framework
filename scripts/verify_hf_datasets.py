#!/usr/bin/env python
"""Verify every pin in the framework's lock files still resolves on the Hugging Face Hub.

A failure means a pinned revision was yanked (e.g. by a force-push) or its dataset was removed
or gated — the affected benchmarks can no longer load their data and the pin needs attention.

Usage::

    uv run python scripts/verify_hf_datasets.py
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
    missing = [
        f"{lockfile.name}: {entry}"
        for lockfile in (HF_REVISIONS_LOCKFILE, FROZEN_HF_REVISIONS_LOCKFILE)
        for entry in HfDatasetRevisions.from_file(lockfile).missing_references(api)
    ]
    if missing:
        raise SystemExit("Unreachable pinned dataset revisions:\n" + "\n".join(missing))
