# Changelog
## 0.2.0 (unreleased)

- Packages are now released to PyPI
- `DEFAULT_FORMATTER` in our models is now a callable, to avoid instantiating formatters at import time
- Import paths in `llm` and `metrics` no longer have a `_llm` and `_metrics` suffix. E.g., `llm/huggingface.py` instead of `llm/huggingface_llm.py`
- Our benchmarks tasks are now registered lazily, which reduces the amount of code that is imported
  at startup time.
- Task look-ups are now insensitive to case, hyphens, underscores and whitespace
- Special tokens are now ignored when computing compression ratios
- Added missing `DOCKER_CODE_EXECUTION` variable to `.env.example`
- Added support for weights and biases + determined pre-emption
- Fixed loading of extra task modules (skip non-evaluation BaseTasks with no NAME attribute), add test that no task with same names get registered
- Fixed `ZERO_SCROLLS_QMSUM` missing from task_names.py
- Fixed BLEU/CHRF/TER min/max scoring when all completions are empty
- Task names in the registry are now enforced to be equal to the class names
- Renamed `ChemBenchMultipleChoice` to `ChemBench` for consistency.
- We've removed all models except those used for testing (they were largely old). The recommended way going forward is to provide
  your own models implementation to the framework.
- Removed and relaxes several main-dependencies
- Moves the generate_task_docs utility to inside the package and add test that documentation is up-to-date
- Fix inconsistent language code for Croatian/Serbian in INCLUDE task

## 0.1.0

- Initial release of `eval-framework`.
