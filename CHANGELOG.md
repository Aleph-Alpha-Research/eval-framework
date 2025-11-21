# Changelog

## Unreleased: 0.2.4-dev

### Models

- Cleaned up `OpenAIModel` class. Those models can now also be evaluated and not only used as judges. Loglikelihood evaluation requests are now implemented (although only supported by a limited number of OpenAI models). Implemented tests for `OpenAIModel` calls. Added concurrency to completion calls
- Added access to Deepseek model API

### Tasks

### Metrics


### General

- Added documentation on `SQUAD` and `SQUAD2` benchmark classes
- Updated documentation on lists of available tasks
- added `.vscode/launch.json`
- Modified the Hendrycks Math task to use the same query template as MATH500 to encourage boxed answer formatting.

## 0.2.3

### Models

- Added `post_process_completion` method to `BaseLLM` class to enable model-specific post-processing of completions before task-specific post-processing is applied.
- The BASELLM class is equiped with `del` call to clear up resources. VLLM and HF APIs offload the respective models off the gpus. OpenAI class disconnects the client.
- Refactored `VLLM` and `HFLLM` interfaces in backwards-compatible way so that there are identical (and flexible!) checkpoint and formatter specification options across VLLM and HFLLM. `VLLMRegistryModel`, `HFLLMRegistryModel`, `HFLLM_from_name` are now deprecated.
- Added `generate_from_samples` method in `BaseLLM` which takes precedence over `generate_from_messages` if implemented.

### Tasks

- `SciQ`: Previously, the benchmark included instructions with context passages that revealed the answer. A new version has been created that removes this context while keeping the original as `SCIQEvalHarness`.
- `TruthfulQA`: Fixed an indexing error that caused the benchmark to return the first correct item instead of the last. Corrected the ground truth for Accuracy to include all label-1 items, rather than only a single item.
- `GSM8K`: In line with the convention of naming the recommended default version as the primary benchmark, `GSM8KLlamaVersion` has been renamed to `GSM8K`, and the original `GSM8K` has been renamed to `GSM8KEvalHarness`.

### Metrics

- `MTBenchJudgePair` and `MTBenchJudgeSingle`: The expected error (KeyError) wouldn't be thrown, resulting in uncaught errors. We now use the same error handling that we do in other tasks.
- Added `ConfidenceWeightedAccuracy`, i.e., the score = probability of the correctly-chosen answer (when it is also the argmax)
- Added `DistributionalCorrectnessScore`, based on Burns (2025) Measuring Language Model Hallucinations Through Distributional Correctness.
- Added `TernaryScore`, based on Kalai et al. (2025) Why language models hallucinate. arXiv:2509.04664.
- `JsonFormat`: added optional `exact_match` score based on whether the generated JSON object equals an expected ground-truth object.

### General

- Added `WANDB_ADDITIONAL_ARTIFACT_REFERENCES` environment variable to reference custom artifacts in W&B.
- Added `resource-cleanup` argument to run.py; enabling a smooth transition in GPU workflows between response generation/evaluation.
- Added `WandbUploader` (for uploading results as W&B artifacts) and refactored `HFUploader` (no change in functionality).
- Config hashes in output directories now do not consider config elements which are irrelevant to actual results.
- Fix: WandB initialization does not crash on overly long model names anymore.
- Fix: "Object of type Role is not JSON serializable" type of errors were fixed.
- Updated examples in the docs to use the updated args and switched default tests to MMLU for more insightful metrics.
- Fix: W&B integration respects WANDB_ARTIFACT_DIR. In addition, new env var WANDB_CACHE_SKIP controls cache use.
- Dropped support for S3 storages without proper SSL certificates.
- Added support for W&B artifacts on local storage which don't need to be downloaded and may be earlier available.
- Fix: `pip install eval_framework[all]` uses uv to fix `ResolveTooDeep` dependency resolver errors.
- Added a CI workflow to test uv and pip installs (CPU only and GPU for VLLM) and avoid trigger with .md changes.
- Updated the CI workflow graph to decouple CPU only test and full test suite with GPU: cpu tests dont wait for docker build.
- Changed implementation of OpenBookQA to be openbook (gives facts in prompt). Old version is available as task OPENBOOKQA_EVAL_HANRESS
- Added a class variable "BYTES_PER_TOKEN" that controls token fertility to allow max_tokens in dataset to be model-specific.
- Changed implementation of OpenBookQA to be openbook (gives facts in prompt). Old version is available as OPENBOOKQA_EVAL_HANRESS task
- Added automated Docker image versioning in release workflow. Docker images are now tagged with `v{major}.{minor}.{patch}`, `v{major}.{minor}`, and `latest` on each release for reproducible deployments.
- Added Docker guide (`docs/docker_guide.md`) for both AA users and external contributors.
- Added template formatting tests to be run by CI.
- Restructured tests to "test_eval_framework" and "tests_template_formatting".

## 0.2.2

### General

- Fix LLM judge not being available via CLI in Determined context

## 0.2.1

### Models

- The `--llm-name` (and `--judge-model-name`) argument can now also be a module path like `eval_framework.llm.huggingface.HFLLM`.
  Combining this with `--llm-args` (`-judge-model-args`) should cover many use-cases without having to provide a `models.py` file.
- Added `eval_framwork.llm.huggingface.HFLLMRegistryModel` and `eval_framwork.llm.vllm.VLLMRegistryModel`
  to conveniently load models from `wandb`.

### Tasks

- Fix for empty `stop_sequences` in `eval_framework.llm.huggingface.StopSequenceCriteria`.
- Fixed dataset loading issues for SQUAD, SQUAD2, FLORES-200, and SPHYR that were causing formatter test failures.
- Pinned `HF_REVISION` for StructEval to `b5512175`, since the train split was renamed test upstream
- Renamed `_get_eval_kwargs` method to `_get_context` in the StructEval task.

### General

- Removed `torch` as a main dependency of `eval_framework`
- Added wandb logging
- Documentation improvements
- Reduced redundant string/path casting

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
- Added accelerate import as default for [transformers] and boto3 in pyproject.toml

## 0.1.0

- Initial release of `eval-framework`.
