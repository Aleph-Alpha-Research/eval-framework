# Controlling Upload Results Guide

This guide explains how to upload evaluation results to HuggingFace repositories and manage your result storage.

## Overview

The upload of evaluation results obtained with `eval-framework` is completely optional and has to be enabled explicitly. When enabled, results are uploaded to a HuggingFace dataset repository for sharing and collaboration.

## Upload Methods

### Method 1: CLI Upload

Add the `--hf-upload-repo` and `--hf-upload-dir` arguments to your CLI command:

```bash
uv run eval_framework \
    --context local \
    --models tests/conftest.py \
    --llm-name Llama31_8B_Instruct_HF \
    --task-name ARC \
    --num-fewshot 3 \
    --num-samples 100 \
    --output-dir "./test_outputs_folder" \
    --hf-upload-repo "my_arbitrary_repo"  # Hugging Face repository
    --hf-upload-dir "my_arbitrary_name"
```

### Method 2: Determined Configuration

Add the `--hf-upload-repo` and `hf_upload_dir` parameter as a hyperparameter in your Determined experiment config:

```yaml
hyperparameters:
  experiment_name: "my_experiment"
  llm_name: "Llama31_8B_Instruct_HF"
  hf_upload_dir: "my_arbitrary_name"  # Note: underscores in YAML
  task_args:
    - task_name: "ARC"
      num_fewshot: 3
      num_samples: 100
```

**Important for Determined**: Due to strict type checking, the `hf_upload_dir` parameter must be present when starting the framework as a Determined experiment. Use an empty string `""` if you don't want to upload results.

## Repository Structure

### Default Repository

By default, results are uploaded to your specified HuggingFace dataset repository.

### Custom Repository

You can specify a different repository with:

```bash
# CLI
--hf-upload-repo "username/repository-name"

# Determined config
hyperparameters:
  hf_upload_repo: "username/repository-name"
```

### Directory Structure

When you specify `--hf-upload-dir "my_arbitrary_name"`, the framework creates:

```
https://huggingface.co/datasets/username/evaluation-results/
└── my_arbitrary_name/
    └── Llama31_8B_Instruct_HF/
        └── v0.1.0_ARC/
            └── fewshot_3__samples_100_91ca2/
                ├── aggregated_results.json
                ├── metadata.json
                ├── output.jsonl
                └── results.jsonl
```

This structure allows different users to maintain their own storage space within the shared repository.


## Authentication

### HuggingFace Token

Ensure you have a valid HuggingFace token with write access:

1. **Get token**: Visit [HuggingFace tokens page](https://huggingface.co/settings/tokens)
2. **Set environment variable**:
   ```bash
   export HUGGINGFACE_HUB_TOKEN="your_token_here"
   ```
3. **Or add to .env file**:
   ```
   HUGGINGFACE_HUB_TOKEN=your_token_here
   ```
4. **Or use HuggingFace CLI login**:
   ```bash
   huggingface-cli login
   ```
   This will prompt you to enter your token interactively and store it securely.
