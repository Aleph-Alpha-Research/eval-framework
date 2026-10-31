# Testing

This repository contains a large and diverse test suite. To keep iteration fast, tests are split into **fast PR tests** and **slow/advanced tests**. Contributors should generally run only the fast tests locally unless reproducing a specific failure.

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

**Recommended local command:**
For most contributors, running the CPU fast tests is sufficient:

```bash
# Run the tests that PR CI runs
uv run pytest -n auto --max-worker-restart=0 -v \
    -m "not cpu_slow and not external_api and not formatter_hash"
````

---

### 2. Advanced Tests

* **Runs on:** manually, when reproducing a CI failure or debugging dependency issues
* **Purpose:** Install all optional extras together, or run the tests that need external services.
* **Includes:**

  * Install tests for all optional extras together
  * Tests marked `external_api` (need credentials for the external services they call)

**Recommended local command (advanced users):**

```bash
uv run --exact pytest -v --noconftest tests/tests_eval_framework/installs/
```

> ⚠️ Warning: The `external_api` tests call real services and need the corresponding credentials.

---

## CI as Source of Truth

The authoritative definition of which tests belong to each tier is encoded in the GitHub workflows:

* `tests.yml` → PR tests, CPU tests, linting

---

## Tips for Contributors

* Run **fast PR tests** before pushing code.
* Do **not** attempt to run the full suite unless reproducing a CI failure.
* CI automatically runs the slow tests on PRs.
