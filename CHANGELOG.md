# Changelog

## [Unreleased]

### Added

- 2025-08-27: Adds support for weights and biases + determined pre-emption

### Changed

- 2025-08-02: remove _get_fewshot_target_text from GSM8KReasoning to make it consistent with other benchmarks that inherit from MATHReasoning and have no few-shot capability such as AIME2024 or MATH500
- 2025-08-29: commented out the flacky SPHYR test

### Fixed

- 2025-08-27: fix loading of extra task modules (skip non-evaluation BaseTasks with no NAME attribute), add test that no task with same names get registered
- 2025-08-27: fix ZERO_SCROLLS_QMSUM missing from task_names.py
- 2025-08-29: fix BLEU/CHRF/TER min/max scoring when all completions are empty

## [0.1.0] - 2025-08-18

Initial release of `eval-framework`.
