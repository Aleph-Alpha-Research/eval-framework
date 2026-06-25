# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

### Setup
- Install dependencies: `uv sync --all-extras`
- Install Python 3.12: `uv python install 3.12 --reinstall` (if needed)
- Install pre-commit hooks: `uv tool install pre-commit && uv run pre-commit install`

### Common Commands
- **Run evaluation (CLI)**: `eval_framework --models src/eval_framework/llm/models.py --llm-name <MODEL> --task-name <TASK> --output-dir <OUTPUT_DIR>`
- **Run tests**: `pytest` or `pytest tests/tests_eval_framework/<test_file>.py` for specific tests
- **Run linting**: `pre-commit run --all-files` (formats code with ruff, checks imports)
- **Type checking**: `mypy` (checks source code in `src/` and `utils/`)

### Project Structure

**Core Architecture**

The eval-framework follows a modular architecture with three main extensible base classes:

1. **BaseLLM** (`src/eval_framework/llm/base.py`) - Language model implementations
   - Used to define how to get responses from models (HuggingFace, OpenAI, custom APIs, etc.)
   - Implementations: `HFLLM`, `OpenAILLM`, `VLLMRunner`, `MistralLLM`, `AlephAlphaLLM`.
   - Requires: `LLM_NAME`, `DEFAULT_FORMATTER`, and methods like `_get_completion()`

2. **BaseTask** (`src/eval_framework/tasks/base.py`) - Benchmark/task definitions
   - Represents evaluation benchmarks (MMLU, HumanEval, InfiniteBench, etc.)
   - New tasks should ideally contain a TASK_STYLER which assembles the prompt from underlying messages. There should be a very good reason to not have this.
   - Three task styles: `COMPLETION`, `LOGLIKELIHOODS`, and LLM-judge-based
   - Requires: `NAME`, `DATASET_PATH`, `SUBJECTS`, and scoring methods
   - Loaded dynamically from `src/eval_framework/tasks/benchmarks/` directory

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

- **`tasks/`**: Task definitions, dataset handling, task loading
  - `base.py`: `BaseTask` class definition
  - `task_loader.py`: Dynamic task discovery and loading
  - `task_style.py`: Enum for task types (completion, loglikelihood, etc.)
  - `benchmarks/`: Individual task implementations (MMLU, HumanEval, etc.)
  - `perturbation.py`: Robustness testing via text perturbations

- **`llm/`**: Language model interfaces
  - `base.py`: `BaseLLM` abstract class
  - `models.py`: Registry of default models
  - Model-specific files for each provider (HuggingFace, OpenAI, etc.)

- **`metrics/`**: Evaluation metrics
  - `base.py`: `BaseMetric` class
  - Subdirectories for metric types (completion, loglikelihood, llm_judge)

- **`template_formatting/`**: Prompt formatting (separate package)
  - `formatter.py`: `Formatter` base class for prompt construction
  - `mistral_formatter.py`: Mistral-specific formatting

- **Context System** (`context/`):
  - `LocalContext`: Run evaluations locally
  - `DeterminedContext`: Distributed evaluation via Determined AI

**Configuration**

- **EvalConfig** (`tasks/eval_config.py`): Main configuration dataclass
  - `task_name`, `task_subjects`: Which task and subjects to evaluate
  - `llm_class`: Which LLM class to instantiate
  - `num_fewshot`, `num_samples`: Evaluation parameters
  - `output_dir`, `wandb_project`: Output and logging settings
  - Supports preemption for distributed runs

## Testing

Run tests with markers to filter by type:
- `pytest -m "not gpu"` - Skip GPU-only tests
- `pytest -m "not slow_download"` - Skip large downloads
- `pytest -m "not external_api"` - Skip external service tests
- `pytest -k <test_name>` - Run specific test by name
- `pytest -n auto` - Run tests in parallel (via pytest-xdist)

Tests use pytest fixtures and mocking. Ensure `WANDB_MODE=disabled` is set in environment (pyproject.toml handles this).

## Development Patterns

**Adding a New Task**
1. Create class inheriting from `BaseTask` in `src/eval_framework/tasks/benchmarks/`
2. Implement required class attributes: `NAME`, `DATASET_PATH`, `SAMPLE_SPLIT`, `FEWSHOT_SPLIT`, `SUBJECTS`
3. If using custom metrics, inherit from `BaseMetric`
4. Task is discovered via registry; add it to `task_names.py` to be discoverable.

**Adding a New Model**
1. Extend `BaseLLM` in `src/eval_framework/llm/`
2. Set `LLM_NAME` and `DEFAULT_FORMATTER`
3. Implement `_get_completion()` or `_get_loglikelihoods()` as needed
4. Register in `models.py` for CLI discovery

**Custom Metrics**
- Inherit from `BaseMetric`
- Implement `compute()` for single samples.
- Supports context objects for additional evaluation data

## Dependencies & Extras

Optional extras (install with `uv sync --all-extras`):
- `transformers`: HuggingFace model evaluation
- `openai`: OpenAI API support
- `vllm`: vLLM inference engine
- `mistral`: Mistral model support
- `api`: Aleph Alpha client
- `determined`: Determined AI integration
- `flash_attn`: Flash attention optimization (handled with build-group flag)

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
