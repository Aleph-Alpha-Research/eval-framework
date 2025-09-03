# Changelog

## [Unreleased]

### Added

- 2025-09-02: Add `DOCKER_CODE_EXECUTION` variable to `.env.example`
- 2025-08-27: Adds support for weights and biases + determined pre-emption

### Changed

- 2025-08-29: commented out the flacky SPHYR test
- 2025-09-03: Our benchmarks tasks are now registered lazily, which reduces the amount of code that is imported
              at startup time.

### Fixed

- 2025-08-27: fix loading of extra task modules (skip non-evaluation BaseTasks with no NAME attribute), add test that no task with same names get registered
- 2025-08-27: fix ZERO_SCROLLS_QMSUM missing from task_names.py
- 2025-08-29: fix BLEU/CHRF/TER min/max scoring when all completions are empty

## [0.1.0] - 2025-08-18

Initial release of `eval-framework`.
