# Changelog

## [Unreleased]

### Added

- 2025-09-02: Add `DOCKER_CODE_EXECUTION` variable to `.env.example`
- 2025-08-27: Adds support for weights and biases + determined pre-emption
- 2025-09-02: move the generate_task_docs utility to inside the package and add tests that documentation is up-to-date

### Changed

- 2025-09-03: Packages are now released to PyPI
- 2025-08-29: commented out the flacky SPHYR test
- 2025-09-03: Our benchmarks tasks are now registered lazily, which reduces the amount of code that is imported
              at startup time.

### Fixed

- 2025-08-27: fix loading of extra task modules (skip non-evaluation BaseTasks with no NAME attribute), add test that no task with same names get registered
- 2025-08-27: fix ZERO_SCROLLS_QMSUM missing from task_names.py
- 2025-08-29: fix BLEU/CHRF/TER min/max scoring when all completions are empty
- 2025-09-02: add pre-commit install to the docs
- 2025-09-02: fix inconsistent language code for Croatian/Serbian in INCLUDE task

## [0.1.0] - 2025-08-18

Initial release of `eval-framework`.
