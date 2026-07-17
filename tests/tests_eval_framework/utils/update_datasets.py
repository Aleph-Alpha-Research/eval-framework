#!/usr/bin/env python
"""Warm the local dataset cache for all registered tasks.

Downloads every task's dataset at its pinned revision (plus the sacrebleu WMT test sets,
which live in their own cache). Already-cached revisions are cheap no-ops, so the same
command serves both a full build and an incremental top-up after a pin change.
"""

from eval_framework.tasks.task_names import make_sure_all_hf_datasets_are_in_cache

if __name__ == "__main__":
    make_sure_all_hf_datasets_are_in_cache()
