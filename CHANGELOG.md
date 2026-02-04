# Changelog

## Main/Unreleased

### Models

### Tasks

### Metrics

### General

### Bug Fixes

## [0.2.12](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.11...v0.2.12) (2026-02-04)


### Features

* add "top_p" param to AlephAlphaAPIModel ([#168](https://github.com/Aleph-Alpha-Research/eval-framework/issues/168)) ([e52c927](https://github.com/Aleph-Alpha-Research/eval-framework/commit/e52c927f293dccce22e5115a4e299e33af6247b1))
* Bump datasets to &gt;=4.0.0 and remove all `trust_remote_code` references. ([#158](https://github.com/Aleph-Alpha-Research/eval-framework/issues/158)) ([c383806](https://github.com/Aleph-Alpha-Research/eval-framework/commit/c38380641302c542bf9222f8a823e13f6df28232))

## [0.2.11](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.10...v0.2.11) (2026-01-30)


### Bug Fixes

* Downloaded w&b artifacts are deleted too early ([#163](https://github.com/Aleph-Alpha-Research/eval-framework/issues/163)) ([157d757](https://github.com/Aleph-Alpha-Research/eval-framework/commit/157d7576330396f7d10731c431892f7e303cf757))
* use aleph-alpha-client concurrency limit and allow &gt;100 concurrent requests ([#166](https://github.com/Aleph-Alpha-Research/eval-framework/issues/166)) ([73b7d97](https://github.com/Aleph-Alpha-Research/eval-framework/commit/73b7d97670fccc82039914ed56cbafa434bb1aba))
* VLLM tokenizer lazy initialization didn't work with W&B ([#165](https://github.com/Aleph-Alpha-Research/eval-framework/issues/165)) ([f38de79](https://github.com/Aleph-Alpha-Research/eval-framework/commit/f38de79a809f0a05e37f1c074569050965c40a7c))

## [0.2.10](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.9...v0.2.10) (2026-01-27)


### Bug Fixes

* prefix dataset paths with hf user id for all tasks that did not have it before ([#160](https://github.com/Aleph-Alpha-Research/eval-framework/issues/160)) ([d5dc178](https://github.com/Aleph-Alpha-Research/eval-framework/commit/d5dc1787325dfeb0cf83e461cf9a81956be7a0ec))

## [0.2.9](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.8...v0.2.9) (2026-01-15)


### Features

* add `repeats` to eval-config ([#150](https://github.com/Aleph-Alpha-Research/eval-framework/issues/150)) ([cb9f860](https://github.com/Aleph-Alpha-Research/eval-framework/commit/cb9f86038f24963199fd5682acc25becb92a0a02))
* add AIME25 benchmark task ([#152](https://github.com/Aleph-Alpha-Research/eval-framework/issues/152)) ([3ef01fc](https://github.com/Aleph-Alpha-Research/eval-framework/commit/3ef01fc1bfa374242e55d5e7c9c6d5d30a379c09))


### Bug Fixes

* docker push on release has one too many 'v's in the tag name ([#153](https://github.com/Aleph-Alpha-Research/eval-framework/issues/153)) ([99e6096](https://github.com/Aleph-Alpha-Research/eval-framework/commit/99e6096e82873e527332fd5c9f386d2d950976d1))

## [0.2.8](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.7...v0.2.8) (2026-01-09)


### Bug Fixes

* normalize math reasoning ([#148](https://github.com/Aleph-Alpha-Research/eval-framework/issues/148)) ([73a8843](https://github.com/Aleph-Alpha-Research/eval-framework/commit/73a88432eaee183ae2274a060e32286bdeda8fa9))
* removed github token from release-please and update image links ([#147](https://github.com/Aleph-Alpha-Research/eval-framework/issues/147)) ([74d59ea](https://github.com/Aleph-Alpha-Research/eval-framework/commit/74d59ea845aed241035199ac87841786d2d75cf5))

## [0.2.7](https://github.com/Aleph-Alpha-Research/eval-framework/compare/v0.2.6...v0.2.7) (2026-01-08)


### Features

* add position randomization for LLM pairwise judges ([#135](https://github.com/Aleph-Alpha-Research/eval-framework/issues/135)) ([e4ed3ec](https://github.com/Aleph-Alpha-Research/eval-framework/commit/e4ed3ec96002becb04f3e1115c04a9a975d1f256))
* added automated documentation through CI and Sphinx ([#127](https://github.com/Aleph-Alpha-Research/eval-framework/issues/127)) ([46ef6b3](https://github.com/Aleph-Alpha-Research/eval-framework/commit/46ef6b34e6608fa38573e87d37f1af7e76d935ae))
* added badges to github readme to link pypi and docs pages ([#139](https://github.com/Aleph-Alpha-Research/eval-framework/issues/139)) ([778bad2](https://github.com/Aleph-Alpha-Research/eval-framework/commit/778bad2ce6b5ee944dc6bed9ce315bc2d68b144f))
* pass AA_TOKEN and AA_INFERENCE_ENDPOINT in the AA model constructor ([#134](https://github.com/Aleph-Alpha-Research/eval-framework/issues/134)) ([93267b6](https://github.com/Aleph-Alpha-Research/eval-framework/commit/93267b60eaf67873277e6d2105900bd890809a55))


### Bug Fixes

* **docs:** resolve broken source links ([#132](https://github.com/Aleph-Alpha-Research/eval-framework/issues/132)) ([c0e37b2](https://github.com/Aleph-Alpha-Research/eval-framework/commit/c0e37b2d32cde341915943bbf3caa45f9d9a6bc5))
* release-please pushes docker to registry and triggers tests ([#138](https://github.com/Aleph-Alpha-Research/eval-framework/issues/138)) ([d291bb4](https://github.com/Aleph-Alpha-Research/eval-framework/commit/d291bb44af2f3576a1a14172c1ab4e7120e0a6d0))


### Documentation

* added documentation for running tests and expected runtimes ([#133](https://github.com/Aleph-Alpha-Research/eval-framework/issues/133)) ([77fd1d3](https://github.com/Aleph-Alpha-Research/eval-framework/commit/77fd1d355f6b6a3c094274d3380cb47e51655971))

## 0.2.6

### Models

### Tasks

### Metrics

### General

- For math reasoning completion, added a finally block that ensures that there is no possibility of the timeout signal going off outside of this block, which crashed the process.

## 0.2.5

### Models
- Move `aleph_alpha.py` to use `/completions` endpoint instead of `/evaluate`. `/evaluate` was just available for model deployed in the luminous workers and is not supported in vllm.

### Tasks

- Added 11 "I don't know" (IDK) task variants: `ARC_IDK`, `COPA_IDK`, `GPQA_IDK`, `HELLASWAG_IDK`, `MMLU_IDK`, `MMLU_PRO_IDK`, `PIQA_IDK`, `OPENBOOKQA_IDK`, `TRUTHFULQA_IDK`, `WINOGENDER_IDK`, and `WINOGRANDE_IDK`. Call for automated hashing.
- Corrected typo in prompt template key for a MTBench LLM-as-a-judge, and implemented tests to ensure these are always what we expect (no typos)

### Metrics

### General
- Updated image urls to be absolute so the pypi page can display them correctly
- Added `llm_judge_prompt` and `llm_judge_response` to MTBENCH metric results

## 0.2.4

### Models

- Cleaned up `OpenAIModel` class. Those models can now also be evaluated and not only used as judges. Loglikelihood evaluation requests are now implemented (although only supported by a limited number of OpenAI models). Implemented tests for `OpenAIModel` calls. Added concurrency to completion calls
- Added access to Deepseek model API

### Tasks

- Added AidanBench benchmark (measures creative divergent thinking by counting unique, coherent responses to open-ended questions) as well as AidanBenchOriginal (the same, but preserving a typo found in the original implementation).

### Metrics

### General

- Added documentation on `SQUAD` and `SQUAD2` benchmark classes
- Updated documentation on lists of available tasks
- Added `.vscode/launch.json`
- Added verbosity levels (0 is critical, 1 is info, 2 is debug) for minimal output
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
