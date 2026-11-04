# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

### Setup
- Install dependencies: `uv sync`
- Install Python 3.12: `uv python install 3.12 --reinstall` (if needed)
- Install pre-commit hooks: `uv tool install pre-commit && uv run pre-commit install`

### Common Commands
- **Run evaluation (CLI)**: `eval_framework --llm-args model_name=<MODEL> [base_url=<URL>] --task-name <TASK> --output-dir <OUTPUT_DIR>` (the CLI always uses `OpenAIModel`; `model_name` is required in `--llm-args` and, for a judge, in `--judge-model-args`)
- **Run tests**: `pytest` or `pytest tests/tests_eval_framework/<test_file>.py` for specific tests
- **Run linting**: `pre-commit run --all-files` (formats code with ruff, checks imports)
- **Type checking**: `mypy` (checks source code in `src/` and `utils/`)

### Project Structure

**Core Architecture**

The eval-framework follows a modular architecture with three main extension points:

1. **BaseLLM** (`src/eval_framework/llm/base.py`) - Language model implementations
   - Used to define how to get responses from models (OpenAI-compatible endpoints, custom APIs, etc.)
   - Implementations: `OpenAIModel` (any OpenAI-compatible endpoint).
   - Requires: `LLM_NAME`, `DEFAULT_FORMATTER`, and methods like `_get_completion()`

2. **ComposedBenchmark** (`src/eval_framework/composed.py`) - Benchmark/task definitions
   - A benchmark is assembled from injected parts via `ComposedBenchmark.compose`: an `EvalKind`
     (the prompt side — `Generative` for free-form answers, `Choice(reader, styler)` for choice
     tasks), an `AnswerPolicy` (the response side — `ExtractFromCompletion` for generation,
     `PickFromCandidates` for loglikelihood scoring), a `FewShotPolicy`, a `DatasetPolicy`
     (loading + revision pinning, e.g. `pinned_by_framework`), and a `SubjectsSelector`
   - Never subclass `ComposedEval`; new behavior goes into the injected parts
   - Benchmark modules live in `src/eval_framework/benchmarks/`, one builder function per variant,
     each module exposing a `*_BENCHMARKS` list

3. **BaseMetric** (`src/eval_framework/metrics/base.py`) - Evaluation metrics
   - Computes scores for task responses (accuracy, BLEU, F1, etc.)
   - Supports both single-sample and aggregated scoring
   - Easily customizable for specialized evaluation needs

**Data Flow**

1. **Response Generation** (`response_generator.py`)
   - Loads dataset samples for selected task
   - Formats samples using task-specific formatters
   - Gets model responses (completion or loglikelihoods)
   - Saves responses to JSON files

2. **Evaluation** (`evaluation_generator.py`)
   - Loads saved responses
   - Applies metrics to compute scores
   - Generates evaluation results with confidence intervals

3. **Result Processing** (`result_processors/`)
   - `ResultsFileProcessor`: Manages output directory structure
   - `HFUploader`: Uploads results to HuggingFace Hub
   - `WandbUploader`: Logs results to Weights & Biases

**Key Modules**

- **`composed.py` / `eval_kind.py` / `answer.py` / `fewshot.py` / `subjects.py` / `choices.py`**: the
  composition vocabulary — benchmark assembly, prompt kinds, answer policies, few-shot policies,
  subject selection, choice readers

- **`benchmarks/`**: Individual benchmark modules (MMLU, HumanEval, GSM8K, etc.)

- **`tasks/`**: Registry, dataset handling, shared task vocabulary
  - `registry.py`: `Registry` of `Benchmark`s, plus `Lazy` for deferred loading
  - `task_names.py`: registers every benchmark module's `*_BENCHMARKS` list (`register_all_tasks`)
  - `dataset_loading.py` / `dataset_revisions.py`: dataset policies and pinned HF revisions (lock files)
  - `task_style.py`: `TaskStyler` strategies for choice-based prompts (`MCStyle`, `ClozeStyle`, `BPBStyle`)
  - `base.py`: shared vocabulary (`Language`, `TaskStyle`, subject-token matching)

- **`llm/`**: Language model interfaces
  - `base.py`: `BaseLLM` abstract class
  - `models.py`: Registry of default models
  - `openai.py`: OpenAI-compatible models

- **`metrics/`**: Evaluation metrics
  - `base.py`: `BaseMetric` class
  - Subdirectories for metric types (completion, loglikelihood, llm_judge)

- **`template_formatting/`**: Prompt formatting (separate package)
  - `formatter.py`: `Formatter` base class for prompt construction
  - `mistral_formatter.py`: Mistral-specific formatting

**Configuration**

- **EvalConfig** (`tasks/eval_config.py`): Main configuration dataclass
  - `task_name`, `task_subjects`: Which task and subjects to evaluate
  - `llm_class`: Which LLM class to instantiate
  - `num_fewshot`, `num_samples`: Evaluation parameters
  - `output_dir`, `wandb_project`: Output and logging settings

## Testing

Run tests with markers to filter by type:
- `pytest -m "not slow_download"` - Skip large downloads
- `pytest -m "not external_api"` - Skip external service tests
- `pytest -k <test_name>` - Run specific test by name
- `pytest -n auto` - Run tests in parallel (via pytest-xdist)

Tests use pytest fixtures and mocking. Ensure `WANDB_MODE=disabled` is set in environment (pyproject.toml handles this).

## Development Patterns

**Adding a New Task**

1. Create a module in `src/eval_framework/benchmarks/` with one builder function per variant returning
   `ComposedBenchmark.compose(...)`; model it on an existing module (e.g. `gsm8k.py` for generative,
   `arc.py` for choice tasks). Name benchmark ids per `docs/naming_convention.md`
   (`{Dataset}_{Source}_{Language}[_{Style}][_{Variant}][_{Subset}]`, required for new tasks)
2. Pin the dataset revision in `tasks/hf-dataset-revisions.json` and use `pinned_by_framework` as the
   dataset policy
3. If using custom metrics, inherit from `BaseMetric`
4. Expose the module's benchmarks as a `*_BENCHMARKS` list and register it in `register_all_tasks`
   (`tasks/task_names.py`); add formatter-hash tests in `tests/tests_eval_framework/benchmarks/`

**Adding a New Model**
1. Extend `BaseLLM` in `src/eval_framework/llm/`
2. Set `LLM_NAME` and `DEFAULT_FORMATTER`
3. Implement `_get_completion()` or `_get_loglikelihoods()` as needed
4. The CLI always uses `OpenAIModel`; use a custom model through the Python API (`EvalConfig(llm_class=...)`)

**Custom Metrics**
- Inherit from `BaseMetric`
- Implement `compute()` for single samples.
- Supports context objects for additional evaluation data

## Conventions

- **Logging**: Use `logging.getLogger(__name__)` for module-level logging
- **File paths**: Use `pathlib.Path` throughout
- **Type hints**: All functions must have type hints; enforced by mypy
- **Imports**: Organized alphabetically by ruff linter
- **Commit messages**: Follow [conventional commits](https://www.conventionalcommits.org/) for automated release notes
- **Line length**: 120 characters (ruff configuration)

## CI/CD & Releases

- Pre-commit hooks run ruff formatting and mypy type checking
- Tests run in CI on multiple Python 3.12 versions
- Releases use [release-please](https://github.com/googleapis/release-please) automation
- Version bumps follow semantic versioning; tag format is `vX.Y.Z`

## Documentation

- Main docs in `docs/` (auto-generated task tables from code)
- Architecture & guides in `docs/*.md` (linked in README)
- Task coverage and metrics listed in `docs/benchmarks_and_metrics.md`
- Results interpretation guide in `docs/understanding_results_guide.html`
