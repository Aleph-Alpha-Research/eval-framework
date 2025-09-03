# Changelog

## 0.2.0 (unreleased)

- Packages are now released to PyPI
- Added missing `DOCKER_CODE_EXECUTION` variable to `.env.example`
- Added support for weights and biases + determined pre-emption
- Fixed loading of extra task modules (skip non-evaluation BaseTasks with no NAME attribute), add test that no task with same names get registered
- Fixed `ZERO_SCROLLS_QMSUM` missing from task_names.py
- Fixed BLEU/CHRF/TER min/max scoring when all completions are empty

## 0.1.0

- Initial release of `eval-framework`.
