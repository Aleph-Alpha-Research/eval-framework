import importlib
import importlib.metadata
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from packaging.requirements import Requirement
from packaging.version import Version


def setup_logging(
    output_dir: Optional[Path] = None, log_level: int = logging.INFO, log_filename: str = "evaluation.log"
) -> logging.Logger:
    """
    Set up centralized logging configuration for the entire framework.

    Args:
        output_dir: Directory to save log files. If None, logs only to console.
        log_level: Logging level (default: INFO)
        log_filename: Name of the log file

    Returns:
        Configured root logger
    """
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    # Console handler (always present)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if output directory provided)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / log_filename

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging configured. File: {log_file}")
    else:
        root_logger.info("Logging configured (console only)")

    print(f"Output directory for logs: {output_dir if output_dir else 'None'}")

    return root_logger


def _validate_package_extras(extras: str | Sequence[str], /, *, package: str = "eval_framework") -> Sequence[str]:
    """Validate that the specified extras are valid for the given package."""
    if isinstance(extras, str):
        extras = [extras]

    metadata = importlib.metadata.metadata(package)
    package_extras = set(metadata.get_all("Provides-Extra") or [])
    for extra in extras:
        if extra not in package_extras:
            raise ValueError(f"Invalid extra: {extra}. Options are {package_extras}")

    return extras


def extra_requires(extra: str, /, *, package: str = "eval_framework") -> list[str]:
    """Return a list of requirements for the specified extra."""
    _validate_package_extras(extra, package=package)
    dist = importlib.metadata.distribution(package)
    requires = dist.requires or []
    extra_str = f"extra == '{extra}'"
    return [r.split(";")[0].strip() for r in requires if r.endswith(extra_str)]


def _dependency_satisfied(dep: str, /) -> bool:
    """Return True if the dependency string is satisfied.

    Args:
        A dependency string: for example "torch~=2.0".
    """
    try:
        dist = importlib.metadata.distribution(Requirement(dep).name)
        installed_version = Version(dist.version)
        req = Requirement(dep)
        return installed_version in req.specifier
    except (importlib.metadata.PackageNotFoundError, Exception):
        return False


def is_extra_installed(extra: str, package: str = "eval_framework") -> bool:
    """Return `True` if all dependencies for a given extra are installed."""
    for req in extra_requires(extra, package=package):
        if not _dependency_satisfied(req):
            return False
    return True
