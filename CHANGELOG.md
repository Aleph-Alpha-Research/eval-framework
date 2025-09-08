# Changelog

## 0.2.0

### Models
- Import paths in `llm` and `metrics` no longer have a `_llm` and `_metrics` suffix. E.g., `llm/huggingface.py` instead of `llm/huggingface_llm.py`.
- We've also removed all models except those used for testing (they were largely old). The recommended way going forward is to provide your own models implementation to the framework.
- `DEFAULT_FORMATTER` in our models is now a callable, to avoid instantiating formatters at import time.

### Tasks
- Our benchmarks tasks are now registered lazily, which reduces the amount of code that is imported
  at startup time. Task look-ups are now insensitive to case, hyphens, underscores and whitespace.
- Task names in the registry are now enforced to be equal to the class names.
- Added `subjects`and `hf_revision` to BaseTask arguments to replace global task re-definition when running with non default values.
- Generate task documentation in `docs/tasks`. Moves the generate_task_docs utility to inside the package and added test that documentation is up-to-date.
- Renamed `ChemBenchMultipleChoice` to `ChemBench` for consistency.
- Fixed `ZERO_SCROLLS_QMSUM` missing from task_names.py
- Fix inconsistent language code for Croatian/Serbian in INCLUDE task

### Metrics
- Fixed BLEU/CHRF/TER min/max scoring when all completions are empty.

### General
- Special tokens are now ignored when computing compression ratios
- Fixed loading of extra task modules (skip non-evaluation BaseTasks with no NAME attribute), add test that no task with same names get registered
- Packages are now released to PyPI
- Removed and relaxes several main-dependencies
- Added support for weights and biases + determined pre-emption
- Added missing `DOCKER_CODE_EXECUTION` variable to `.env.example`

## 0.1.0

- Initial release of `eval-framework`.
