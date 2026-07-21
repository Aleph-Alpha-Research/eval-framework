#!/usr/bin/env python
"""Refresh every pin in ``hf-dataset-revisions.json`` to its dataset's latest commit SHA.

The lock file's keys define which datasets are pinned; refreshing only updates their values.
Hand-maintained lock files (e.g. ``frozen-hf-dataset-revisions.json``) are never touched.

Usage::

    uv run python scripts/update_hf_datasets.py
"""

import logging

from huggingface_hub import HfApi

from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE, HfDatasetRevisions

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    revisions = HfDatasetRevisions.from_file(HF_REVISIONS_LOCKFILE)
    revisions.update_to_latest(HfApi())
    revisions.to_file(HF_REVISIONS_LOCKFILE)
