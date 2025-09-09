import importlib
import importlib.metadata
from collections.abc import Sequence

from packaging.requirements import Requirement
from packaging.version import Version


def validate_package_extras(extras: str | Sequence[str], /, *, package: str = "eval_framework") -> Sequence[str]:
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
    validate_package_extras(extra, package=package)
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
