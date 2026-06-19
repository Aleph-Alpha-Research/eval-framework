"""Naming-convention check for registered tasks.

Task class names follow the grammar in ``docs/task_naming.md``::

    {Dataset}_{Source}_{Language}[_{Style}][_{Variant}][_{Subset}]
"""

from eval_framework.tasks.registry import registered_task_names
from eval_framework.tasks.task_names import parse_task_name


def _follows_naming_convention(name: str) -> bool:
    """Return ``True`` if a task name matches the canonical grammar."""
    return parse_task_name(name) is not None


def test_registered_tasks_follow_naming_convention() -> None:
    """Every registered task follows the grammar (see docs/task_naming.md)."""
    offenders = sorted(name for name in registered_task_names() if not _follows_naming_convention(name))
    assert not offenders, (
        f"{len(offenders)} task name(s) do not follow docs/task_naming.md: {offenders}. "
        "Rename them to {Dataset}_{Source}_{Language}[_{Style}][_{Variant}][_{Subset}]."
    )


def _report() -> str:
    names = sorted(registered_task_names())
    conforming = [name for name in names if _follows_naming_convention(name)]
    nonconforming = [name for name in names if not _follows_naming_convention(name)]

    lines = [f"Conforming ({len(conforming)}):"]
    lines.extend(f"  + {name}" for name in conforming)
    if not conforming:
        lines.append("  (none yet)")
    lines.append("")
    lines.append(f"Not yet conforming ({len(nonconforming)}):")
    lines.extend(f"  - {name}" for name in nonconforming)
    if not nonconforming:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"Tasks: {len(conforming)}/{len(names)} follow the naming convention.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_report())
