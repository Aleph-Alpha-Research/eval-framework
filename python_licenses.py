#!/usr/bin/env python3
"""Generate third-party dependency + license reports for Python (uv) projects.

Thin wrapper around `pip-licenses` plus the boring glue:
  - SPDX normalization for free-form classifier strings
  - Internal/first-party package filtering
  - Runtime-only filtering: dev deps excluded by walking `uv.lock`

Usage:

    uv run python_licenses.py deps        # writes THIRD_PARTY.md
    uv run python_licenses.py licenses    # writes THIRD_PARTY_LICENSES.md
    uv run python_licenses.py both        # do both

Requirements:

  - `pip-licenses` in the project venv:
        uv add --dev pip-licenses && uv sync
  - Committed `uv.lock` at repo root.

Configuration (optional, in pyproject.toml). The script reads
`[tool.licenses.python]` if present, otherwise `[tool.licenses]`:

    [tool.licenses.python]
    internal-packages = ["my-internal-pkg"]
    deps-output = "THIRD_PARTY.md"
    licenses-output = "THIRD_PARTY_LICENSES.md"

    [tool.licenses.python.spdx-overrides]
    FastWARC = "Apache-2.0"      # for packages with garbled metadata
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404  # subprocess used with controlled input
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_DEPS_OUTPUT = "THIRD_PARTY.md"
DEFAULT_LICENSES_OUTPUT = "THIRD_PARTY_LICENSES.md"

# Free-form license strings emitted by pip-licenses (from PyPI classifiers,
# which predate SPDX) mapped to canonical SPDX identifiers.
SPDX_BY_PIP_LICENSE: dict[str, str] = {
    "Apache 2.0": "Apache-2.0",
    "Apache License 2.0": "Apache-2.0",
    "Apache Software License": "Apache-2.0",
    "Apache-2.0": "Apache-2.0",
    "Apache Software License; BSD License": "Apache-2.0 OR BSD-3-Clause",
    "MIT": "MIT",
    "MIT License": "MIT",
    "MIT-0": "MIT-0",
    "BSD": "BSD-3-Clause",
    "BSD License": "BSD-3-Clause",
    "3-Clause BSD License": "BSD-3-Clause",
    "BSD-3-Clause": "BSD-3-Clause",
    "BSD-2-Clause": "BSD-2-Clause",
    "ISC License (ISCL)": "ISC",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "Python Software Foundation License": "PSF-2.0",
    "PSF-2.0": "PSF-2.0",
}

_PEP508_NAME = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\s*\[[^\]]*\])?\s*(.*)$")


# --------------------------------------------------------------------------- #
# Configuration.
# --------------------------------------------------------------------------- #


class Config:
    repo_root: Path
    pyproject: dict[str, Any]
    project_name: str
    internal_packages: frozenset[str]
    spdx_overrides: dict[str, str]
    deps_output: Path
    licenses_output: Path

    def __init__(self, repo_root: Path) -> None:
        pyproject_path = repo_root / "pyproject.toml"
        if not pyproject_path.is_file():
            raise SystemExit(f"No pyproject.toml at {pyproject_path}.")
        self.repo_root = repo_root
        self.pyproject = tomllib.loads(pyproject_path.read_text())

        base = self.pyproject.get("tool", {}).get("licenses", {})
        cfg: dict[str, Any] = {**base, **base.get("python", {})}

        self.project_name = self.pyproject.get("project", {}).get("name", "")
        internal = set(cfg.get("internal-packages", []))
        internal.add(self.project_name)
        self.internal_packages = frozenset(p for p in internal if p)
        self.spdx_overrides = dict(cfg.get("spdx-overrides", {}))
        self.deps_output = repo_root / cfg.get("deps-output", DEFAULT_DEPS_OUTPUT)
        self.licenses_output = repo_root / cfg.get("licenses-output", DEFAULT_LICENSES_OUTPUT)


# --------------------------------------------------------------------------- #
# Dependency walk via uv.lock.
# --------------------------------------------------------------------------- #


def _direct_deps(cfg: Config) -> list[tuple[str, str]]:
    """``(name, version-spec)`` tuples from ``[project.dependencies]``."""
    out: list[tuple[str, str]] = []
    for raw in cfg.pyproject.get("project", {}).get("dependencies", []):
        match = _PEP508_NAME.match(raw.strip())
        if not match:
            continue
        name, spec = match.group(1), match.group(2).strip()
        if name in cfg.internal_packages:
            continue
        out.append((name, spec))
    return out


def _runtime_packages(cfg: Config) -> dict[str, str]:
    """``{name -> version}`` for all runtime transitive Python deps."""
    lock_path = cfg.repo_root / "uv.lock"
    if not lock_path.is_file():
        raise SystemExit(
            f"No uv.lock at {lock_path}; this script needs a committed lockfile "
            "to scope licenses to runtime deps. Run `uv lock` first."
        )
    with lock_path.open("rb") as fh:
        lock: dict[str, Any] = tomllib.load(fh)

    project = cfg.project_name.lower()
    canonical_internal = {n.lower() for n in cfg.internal_packages}
    by_name = {pkg.get("name", "").lower(): pkg for pkg in lock.get("package", [])}

    root = by_name.get(project)
    if root is None:
        raise SystemExit(f"Project {project!r} not in uv.lock; lockfile may be stale.")

    visited: dict[str, str] = {}
    stack: list[str] = [dep["name"].lower() for dep in root.get("dependencies", []) if "name" in dep]
    while stack:
        pkg = stack.pop()
        if pkg in visited or pkg == project or pkg in canonical_internal:
            continue
        entry = by_name.get(pkg)
        if entry is None:
            continue
        visited[pkg] = entry.get("version", "(unknown)")
        for dep in entry.get("dependencies", []):
            dep_name = dep.get("name", "")
            if dep_name:
                stack.append(dep_name.lower())
    return visited


# --------------------------------------------------------------------------- #
# License collection via pip-licenses.
# --------------------------------------------------------------------------- #


def _normalize_spdx(raw: str, package: str, overrides: dict[str, str]) -> str:
    if package in overrides:
        return overrides[package]
    # Some wheels stuff the entire license body into the License classifier;
    # trim before lookup so the SPDX table can still match the header.
    head = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    if len(head) > 80:
        head = head[:80]
    return SPDX_BY_PIP_LICENSE.get(head, head or "UNKNOWN")


def _project_python(cfg: Config) -> str:
    """Path to the project venv's interpreter (where pip-licenses lives)."""
    for venv in (cfg.repo_root / ".venv", cfg.repo_root / "venv"):
        unix = venv / "bin" / "python"
        if unix.is_file():
            return str(unix)
        win = venv / "Scripts" / "python.exe"
        if win.is_file():
            return str(win)
    return sys.executable


def _check_pip_licenses(python: str) -> None:
    result = subprocess.run(  # nosec  # hardcoded args
        [python, "-c", "import piplicenses"],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return
    raise SystemExit(
        f"pip-licenses is required but not installed in {python}.\n"
        "Install it into the project venv:\n"
        "  uv add --dev pip-licenses && uv sync"
    )


def _collect_licenses(cfg: Config) -> list[dict[str, str]]:
    runtime = _runtime_packages(cfg)
    if not runtime:
        return []
    python = _project_python(cfg)
    _check_pip_licenses(python)
    output = subprocess.check_output(  # nosec  # hardcoded args
        [
            python,
            "-m",
            "piplicenses",
            "--from=mixed",
            "--format=json",
            "--with-urls",
            "--with-license-file",
            "--no-license-path",
            "--packages",
            *sorted(runtime.keys()),
        ],
        cwd=cfg.repo_root,
        text=True,
    )
    entries: list[dict[str, str]] = json.loads(output)
    for entry in entries:
        entry["SPDX"] = _normalize_spdx(entry.get("License", ""), entry["Name"], cfg.spdx_overrides)
    return entries


# --------------------------------------------------------------------------- #
# Markdown rendering.
# --------------------------------------------------------------------------- #


def _format_deps(direct: list[tuple[str, str]], transitive: list[tuple[str, str]]) -> str:
    lines = [
        "# Third-Party Dependencies (Python)",
        "",
        "Auto-generated by `python_licenses.py deps`. Dev/test dependencies",
        "and first-party/internal packages are excluded.",
        "",
        f"- {len(direct)} direct, {len(transitive)} transitive",
        "",
        "## Direct",
        "",
        "| Package | Constraint |",
        "|---|---|",
    ]
    for name, spec in direct:
        lines.append(f"| `{name}` | `{spec or '(any)'}` |")
    lines.extend(["", "## Transitive", "", "| Package | Version |", "|---|---|"])
    for name, version in transitive:
        lines.append(f"| `{name}` | `{version}` |")
    return "\n".join(lines) + "\n"


def _format_licenses(entries: list[dict[str, str]]) -> str:
    lines = [
        "# Third-Party Licenses (Python)",
        "",
        "Auto-generated by `python_licenses.py licenses`. Covers all transitive",
        "runtime Python dependencies; dev-only and internal packages are excluded.",
        "",
    ]
    counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        counts[entry["SPDX"]] += 1
    lines.extend(["## Summary", "", "| License (SPDX) | Count |", "|---|---|"])
    for lic, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{lic}` | {count} |")
    lines.append("")

    by_license: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        by_license[entry["SPDX"]].append(entry)
    for lic in sorted(by_license):
        lines.extend([f"## {lic}", ""])
        for entry in sorted(by_license[lic], key=lambda e: e["Name"].lower()):
            url = entry.get("URL", "")
            suffix = f" - {url}" if url and url != "UNKNOWN" else ""
            lines.append(f"- **{entry['Name']}** {entry['Version']}{suffix}")
        lines.append("")

    lines.extend(["## Full license texts", ""])
    for entry in sorted(entries, key=lambda e: e["Name"].lower()):
        text = (entry.get("LicenseText") or "").strip()
        if not text or text == "UNKNOWN":
            continue
        lines.extend(
            [
                f"### {entry['Name']} {entry['Version']} ({entry['SPDX']})",
                "",
                "```text",
                text,
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Top-level commands.
# --------------------------------------------------------------------------- #


def run_deps(cfg: Config) -> None:
    direct = sorted(
        ((n, s or "(any)") for n, s in _direct_deps(cfg)),
        key=lambda d: d[0].lower(),
    )
    direct_names = {n.lower() for n, _ in direct}
    runtime = _runtime_packages(cfg)
    transitive = sorted(
        ((n, v) for n, v in runtime.items() if n not in direct_names),
        key=lambda d: d[0].lower(),
    )
    cfg.deps_output.write_text(_format_deps(direct, transitive))
    print(
        f"=== Wrote {cfg.deps_output.relative_to(cfg.repo_root)} "
        f"({len(direct)} direct + {len(transitive)} transitive) ==="
    )


def run_licenses(cfg: Config) -> None:
    print("Collecting Python runtime licenses...", file=sys.stderr)
    entries = _collect_licenses(cfg)
    cfg.licenses_output.write_text(_format_licenses(entries))
    print(f"=== Wrote {cfg.licenses_output.relative_to(cfg.repo_root)} ({len(entries)} packages) ===")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Python third-party deps + license report.")
    parser.add_argument(
        "command",
        nargs="?",
        default="both",
        choices=("deps", "licenses", "both"),
        help="What to generate (default: both).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing pyproject.toml (default: cwd).",
    )
    args = parser.parse_args(argv)

    cfg = Config(args.repo_root.resolve())
    if args.command in ("deps", "both"):
        run_deps(cfg)
    if args.command in ("licenses", "both"):
        run_licenses(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
