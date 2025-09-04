# Changelog
## 0.2.0 (unreleased)

- Packages are now released to PyPI
- `DEFAULT_FORMATTER` in our models is now a callable, to avoid instantiating formatters at import time
- Import paths in `llm` and `metrics` no longer have a `_llm` and `_metrics` suffix. E.g., `llm/huggingface.py` instead of `llm/huggingface_llm.py`
- Our benchmarks tasks are now registered lazily, which reduces the amount of code that is imported
  at startup time.
- Task look-ups are now insensitive to case, hyphens, underscores and whitespace
- Added missing `DOCKER_CODE_EXECUTION` variable to `.env.example`
- Added support for weights and biases + determined pre-emption
- Fixed loading of extra task modules (skip non-evaluation BaseTasks with no NAME attribute), add test that no task with same names get registered
- Fixed `ZERO_SCROLLS_QMSUM` missing from task_names.py
- Fixed BLEU/CHRF/TER min/max scoring when all completions are empty
- Task names in the registry are now enforced to be equal to the class names
- Renamed `ChemBenchMultipleChoice` to `ChemBench` for consistency.

## 0.1.0

- Initial release of `eval-framework`.
