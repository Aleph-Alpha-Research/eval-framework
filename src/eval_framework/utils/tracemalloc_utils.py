"""Helpers for tracemalloc-based allocation snapshots (use with logging level DEBUG)."""

from __future__ import annotations

import logging
import tracemalloc

_module_logger = logging.getLogger(__name__)
_tracemalloc_start_failed = False

# Log tracemalloc snapshots every N samples/iterations in long-running loops (DEBUG only).
TRACEMALLOC_PROGRESS_INTERVAL = 500


def ensure_tracemalloc_started(*, nframe: int = 25) -> bool:
    """Start tracing if not already active. Returns False if tracing is unavailable."""
    global _tracemalloc_start_failed
    if _tracemalloc_start_failed:
        return False
    if tracemalloc.is_tracing():
        return True
    try:
        tracemalloc.start(nframe)
        return True
    except Exception as e:
        _tracemalloc_start_failed = True
        _module_logger.warning(
            "tracemalloc: could not start tracing; further tracemalloc debug snapshots will be skipped (%s)",
            e,
        )
        return False


def log_tracemalloc_debug(
    logger: logging.Logger,
    event: str,
    *,
    top_stats: int = 8,
) -> None:
    """
    Log current/peak traced Python allocation since tracing started, and top allocation sites.

    No-ops quickly when DEBUG is disabled for this logger, to avoid snapshot overhead.
    Never raises: failures are logged at WARNING on the given logger.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if not ensure_tracemalloc_started():
        return
    try:
        current, peak = tracemalloc.get_traced_memory()
        logger.debug(
            "tracemalloc [%s]: current=%d bytes (%.2f MiB), peak=%d bytes (%.2f MiB)",
            event,
            current,
            current / (1024 * 1024),
            peak,
            peak / (1024 * 1024),
        )
        if top_stats <= 0:
            return
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics("lineno")
        for i, stat in enumerate(stats[:top_stats], 1):
            logger.debug("tracemalloc [%s] top_alloc %d: %s", event, i, stat)
    except Exception as e:
        logger.warning("tracemalloc [%s]: could not collect stats (%s)", event, e)
