# Testing

This repository contains a large and diverse test suite. To keep iteration fast, tests are split into **fast PR tests**, **slow/advanced tests**, and **nightly workflows**. Contributors should generally run only the fast tests locally unless reproducing a specific failure.

---

## Test Tiers

### 1. Fast / PR Tests
- **Runs on:** Every push to `main`, pull requests, and merge queue.
- **Runtime:** ~20 minutes total
- **Purpose:** Ensure code correctness for PRs without running the heaviest tests.
- **Includes:**
  - Linting, pre-commit, type checks (2 min)
  - Tag setup and HuggingFace datasets cache pull (2 min)
  - Docker image build (5 min)
  - UV install dependency tests (1 min)
  - CPU tests excluding slow/external tests (3 min)
  - CPU slow tests (3–4 min)
  - Formatter hash tests (3 min)
  - GPU tests / optional extras (12 min)

**Recommended local command:**
For most contributors, running the CPU fast tests is sufficient:

```bash
# Run the tests that PR CI runs
uv run --all-extras pytest -n auto --max-worker-restart=0 -v \
    -m "not gpu and not cpu_slow and not external_api and not formatter_hash"
````

---

### 2. Advanced / GPU Tests

* **Runs on:** PR workflow (`test-docker-gpu`)
* **Runtime:** ~12 min
* **Purpose:** Run GPU tests or all optional extras together. Typically only required if debugging GPU-specific issues.
* **Includes:**

  * GPU tests excluding CPU-slow / external API / vllm (12 min)
  * Optional extras (`vllm`, `mistral`) installations

**Recommended local command (advanced users):**

```bash
uv run --exact --all-extras pytest -v --noconftest tests/tests_eval_framework/installs/
```

> ⚠️ Warning: Running GPU/full extras locally may take significant time and requires a GPU.

---

### 3. Nightly Workflows

* **Nightly HuggingFace dataset cache rebuild**

  * **Runtime:** ~20 min
  * **Purpose:** Rebuild the full dataset cache for CI and experiments
  * **Command:** `uv run --extra=comet --extra=openai python tests/tests_eval_framework/utils/update_datasets.py rebuild`

* **Nightly Docker build cache**

  * **Runtime:** ~30 sec
  * **Purpose:** Refresh Docker build cache for PR workflows

> Nightly workflows are **not expected to be run locally**. They ensure CI has up-to-date datasets and Docker cache.

---

## CI as Source of Truth

The authoritative definition of which tests belong to each tier is encoded in the GitHub workflows:

* `tests.yml` → PR tests, CPU and GPU tests, linting
* `nightly_hf_cache_build.yml` → full dataset cache rebuild
* `nightly_docker_cache_build.yml` → Docker cache refresh

---

## Tips for Contributors

* Run **fast PR tests** before pushing code.
* Do **not** attempt to run the full suite unless reproducing a nightly/CI failure.
* CI automatically runs GPU and slow tests on PRs; nightly workflows cover the rest.
