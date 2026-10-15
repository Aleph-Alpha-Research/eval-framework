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

The eval-framework follows a modular architecture built around three extension points:

1. **BaseLLM** (`src/eval_framework/llm/base.py`) - Language model implementations
   - Used to define how to get responses from models (OpenAI-compatible endpoints, custom APIs, etc.)
   - Implementations: `OpenAIModel` (any OpenAI-compatible endpoint).
   - Requires: `generate_from_messages()` (completions) and `logprobs()` (loglikelihoods)

2. **Benchmark / Eval** (`src/eval_framework/contract.py`) - Benchmark definitions
   - A `Benchmark` is an immutable, registered description of a benchmark (id, metrics, subjects, docs). Its
     `create(num_fewshot, custom_subjects, custom_hf_revision, seed)` is a factory for an `Eval`: one concrete
     run that yields `Sample`s and generates completions.
   - Benchmarks are built with `ComposedBenchmark` (`src/eval_framework/composed.py`), which plugs together
     independent parts instead of subclassing:
     - `DatasetPolicy` (`tasks/dataset_loading.py`, `tasks/dataset_revisions.py`): where the data comes from,
       e.g. `pinned_by_framework("allenai/ai2_arc")`, `.subset(...)`, `.subject_encoded_in_column(...)`
     - `SubjectsSelector` (`subjects.py`): which slices to evaluate, `NoSubject` (default) or `ListOfSubjects`
     - `EvalKind` (`eval_kind.py`): the prompt side — messages, candidates, ground truth, scoring metrics.
       `Choice(reader, styler)` for choice-based tasks, `Generative(...)` for free-form ones
     - `AnswerPolicy` (`answer.py`): the answer side — response type, stop sequences, max tokens, extraction.
       `PickFromCandidates`, `ExtractFromCompletion` or `ReconstructProgram`
     - `FewShotPolicy` (`fewshot.py`): `FewShot(source, renderer)` or `NoFewShot`. Sources: `SampleSplit`
       (eval split, leak-safe), `FewShotSplit(split)`, `Predefined(items, ...)`. Renderers: `ChoiceRenderer`,
       `FunctionRenderer`
     - `ChoiceReader` (`choices.py`): the only place that knows a dataset's item schema (choice tasks)
     - `TaskStyler` (`tasks/task_style.py`): how a choice prompt is presented and scored — `MCStyle`,
       `ClozeStyle`, `BPBStyle`, and `.with_abstention_option(...)`. Only used by choice-based tasks; the
       response type comes from the `AnswerPolicy`, not the styler
   - Response types (`ResponseType`): `COMPLETION` and `LOGLIKELIHOODS`. LLM-judge scoring is a metric choice,
     not a separate type.
   - `BaseTask` (`tasks/base.py`) is legacy: no built-in benchmark uses it. It is still accepted for user plugins
     via `register_task`, which wraps it in an `Eager` benchmark.

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

- **Top-level benchmark modules** (`src/eval_framework/`)
  - `contract.py`: `Benchmark`, `Eval`, `Sample`, `ResponseType`
  - `composed.py`: `ComposedBenchmark` and `ComposedEval`
  - `eval_kind.py`, `answer.py`, `fewshot.py`, `subjects.py`, `choices.py`: the composable parts (see above)
  - `benchmarks/`: Individual benchmark implementations (ARC, MMLU, GSM8K, HumanEval, etc.)

- **`tasks/`**: Dataset handling, registration, legacy task base
  - `registry.py`: The registry of `Benchmark`s, keyed by `id()`
  - `task_names.py`: Registers all built-in benchmarks
  - `task_loader.py`: Loads user plugin modules from a directory
  - `task_style.py`: `TaskStyler` classes for choice-based prompts
  - `dataset_loading.py`, `dataset_revisions.py`: `DatasetPolicy` implementations and pinned HF revisions
  - `base.py`: Legacy `BaseTask`, plus shared enums (`TaskStyle`, `Language`)

- **`llm/`**: Language model interfaces
  - `base.py`: `BaseLLM` abstract class
  - `models.py`: Default models file (re-exports `OpenAIModel`)
  - `openai.py`: OpenAI-compatible models

- **`metrics/`**: Evaluation metrics
  - `base.py`: `BaseMetric` class
  - Subdirectories for metric types (completion, loglikelihood, llm_judge)

- **`template_formatting/`**: Prompt formatting (separate package)
  - `formatter.py`: `BaseFormatter` base class for prompt construction
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
- `pytest -m "not slow_download"` - Skip large downloads
- `pytest -m "not external_api"` - Skip external service tests
- `pytest -k <test_name>` - Run specific test by name
- `pytest -n auto` - Run tests in parallel (via pytest-xdist)

Tests use pytest fixtures and mocking. Ensure `WANDB_MODE=disabled` is set in environment (pyproject.toml handles this).

## Development Patterns

**Adding a New Benchmark**

Do not subclass `BaseTask`. Write a module in `src/eval_framework/benchmarks/` with factory functions that return a
`ComposedBenchmark`:

1. **Choice-based** (multiple choice, cloze, BPB): write a `ChoiceReader` for the item schema and call
   `ComposedBenchmark.choice(id=..., reader=..., styler=..., sample_split=..., fewshot_split=..., subjects=...,
   dataset_policy=..., language=...)`. The same reader + styler render the few-shot demos. Reference: `arc.py`.
2. **Generative** (free-form answers, code): call `ComposedBenchmark.compose(...)` with a `Generative(...)` kind, an
   `AnswerPolicy` (e.g. `ExtractFromCompletion`) and a `FewShotPolicy`. Reference: `gsm8k.py`.
3. Variants differ only in the arguments they pass (e.g. the styler); share a private helper rather than copying.
   If a task doesn't fit `Choice` or `Generative`, write a new `EvalKind` (see `cot.py`, `winogrande.py`) instead
   of adding options to `Generative`.
4. Collect the module's benchmarks in a `<NAME>_BENCHMARKS: list[Benchmark]` constant and register it in
   `tasks/task_names.py` (a `register_<name>_tasks` function calling `registry.add(...)`, invoked from
   `register_all_tasks`).
5. If using custom metrics, inherit from `BaseMetric`.

**Adding a New Model**
1. Extend `BaseLLM` in `src/eval_framework/llm/`
2. Implement `generate_from_messages()` and/or `logprobs()` as needed
3. Register in `models.py` for CLI discovery

**Custom Metrics**
- Inherit from `BaseMetric`
- Implement `calculate()` for single responses.
- Supports context objects for additional evaluation data

## Dependencies & Extras

Optional extras (install with `uv sync --all-extras`):
- `determined`: Determined AI integration

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
- Results interpretation guide in `docs/understanding_results_guide.md`
