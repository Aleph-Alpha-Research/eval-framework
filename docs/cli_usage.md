# Using CLI

The Eval-Framework CLI provides a flexible interface for evaluating LLMs across a wide range of benchmarks. Whether you're running evaluations locally or in a distributed environment, the CLI allows you to configure tasks, models, and metrics with ease.

## Quick Start

Install the package:

```
uv sync --all-extras
```

And execute a single evaluation locally:

```bash
eval_framework \
    --models src/eval_framework/llm/models.py \
    --llm-name Smollm135MInstruct \
    --task-name "MMLU" \
    --task-subjects "abstract_algebra" "anatomy" \
    --output-dir ./eval_results \
    --num-fewshot 5 \
    --num-samples 10
```

## Command Structure

```bash
uv run eval_framework [OPTIONS]
```

### Required Arguments

**`--llm-name LLM_NAME`**
Either a module path to a model, or the name of a model found in the file provided via the `--models` flag.

### Execution Configuration

**`--models MODELS`**
Path to the Python module file containing model classes.

**`--llm-args [LLM_ARGS ...]`**
Arguments to pass to the LLM as key=value pairs.

**`--task-name TASK_NAME`**
The name of the task to evaluate.

**`--output-dir OUTPUT_DIR`**
The path for evaluation outputs.

**`--num-samples NUM_SAMPLES`**
The number of samples per subject to evaluate.

**`--num-fewshot NUM_FEWSHOT`**
The number of fewshot examples to use.

**`--max-tokens`**
The maximum number of tokens to generate for each sample. Overwrites any task default value.

**`--batch-size BATCH_SIZE`**
Size of batch of samples to send to the LLM for evaluation in parallel. Use 1 for sequential running (default).

### Task Configuration

**`--task-subjects TASK_SUBJECTS [TASK_SUBJECTS ...]`**
The subjects of the task to evaluate. If empty, all subjects are evaluated. Subjects in the form of tuples can be specified in a comma-delimited way, possibly using wildcard * in some dimensions of a tuple.
Examples: `"DE_DE, *"` or `"FR_FR, astronomy"`

**`--hf-revision HF_REVISION`**
A tag name, a branch name, or commit hash for the task HF dataset.

### Judge Models

**`--judge-models JUDGE_MODELS`**
The path to the Python module file containing LLM judge model classes.

**`--judge-model-name JUDGE_MODEL_NAME`**
The class derived from `eval_framework.llm.base.BaseLLM` found in the judge-models module to instantiate for LLM judge evaluation metrics.

**`--judge-model-args JUDGE_MODEL_ARGS`**
The args of the judge model used.

### Perturbations

**`--perturbation-type TYPE`**
The type of perturbation to apply to task instructions. Note that this may not make sense for some prompts for example, those containing math and code.

**`--perturbation-probability PROBABILITY`**
The probability of applying a perturbation to each word or character (between 0.0 and 1.0).

**`--perturbation-seed SEED`**
Random seed controlling perturbations.

### Logging & Tracking

**`--wandb-project WANDB_PROJECT`**
The name of the Weights & Biases project to log runs to.

**`--wandb-entity WANDB_ENTITY`**
The name of the Weights & Biases entity to log runs to. Defaults to the user's default entity.

**`--wandb-run-id WANDB_RUN_ID`**
The ID of an existing Weights & Biases run to resume. If not given, creates a new run. If given and exists, will continue the run but will overwrite the python command logged in WandB.

**`--wandb-upload-results` or `--no-wandb-upload-results`**
Whether to upload results as an artifact to Weights & Biases (default: True). Needs `--wandb-project` to be set.

**`--description DESCRIPTION`**
Description of the run. This will be added to the metadata of the run to help with bookkeeping.

**`--verbosity VERBOSITY`**
Set the logging verbosity level: 0=CRITICAL, 1=INFO, 2=DEBUG.

### Environment

**`--context {local,determined}`**
The context in which the evaluation is run.

**`-h, --help`**
Show help message and exit.

## Running Hugging Face Models

You can run models directly from Hugging Face Hub using the `HFLLM` class:

```bash
uv run eval_framework \
    --llm-name 'eval_framework.llm.huggingface.HFLLM' \
    --llm-args model_name="microsoft/DialoGPT-medium" formatter_name="Llama3Formatter" \
    --task-name "MMLU" \
    --task-subjects "abstract_algebra" \
    --output-dir ./eval_results \
    --num-fewshot 5 \
    --num-samples 10
```

This approach allows you to evaluate any model available on Hugging Face by specifying the `model_name` and appropriate `formatter_name` in the `--llm-args` parameter.

## Configuring Sampling Parameters for vLLM Models

vLLM models support configurable sampling parameters through the `--llm-args` parameter. You can specify individual sampling parameters using dot notation:

```bash
uv run eval_framework \
    --llm-name 'eval_framework.llm.models.Qwen3_0_6B_VLLM' \
    --llm-args sampling_params.temperature=0.7 sampling_params.top_p=0.95 sampling_params.max_tokens=150 \
    --task-name "MMLU" \
    --task-subjects "abstract_algebra" \
    --output-dir ./eval_results \
    --num-fewshot 5 \
    --num-samples 10
```

You can also combine sampling parameters with other model arguments:

```bash
uv run eval_framework \
    --llm-name 'eval_framework.llm.models.Qwen3_0_6B_VLLM' \
    --llm-args max_model_len=2048 sampling_params.temperature=0.8 sampling_params.top_p=0.9 \
    --task-name "MMLU" \
    --task-subjects "abstract_algebra" \
    --output-dir ./eval_results \
    --num-fewshot 5 \
    --num-samples 10
```
