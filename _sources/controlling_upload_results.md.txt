# Controlling HuggingFace Upload Results Guide

This guide explains how to upload evaluation results to HuggingFace repositories and manage your result storage.

## Overview

The upload of evaluation results obtained with `eval-framework` is completely optional and has to be enabled explicitly. When enabled, results are uploaded to a HuggingFace dataset repository for sharing and collaboration.

## Upload Methods

### Method 1: CLI Upload

Add the `--hf-upload-repo` and `--hf-upload-dir` arguments to your CLI command:

```bash
uv run eval_framework \
    --llm-args model_name="gpt-4o-mini" \
    --task-name ARC \
    --num-fewshot 3 \
    --num-samples 100 \
    --output-dir "./eval_results" \
    --hf-upload-repo "my_arbitrary_repo"  # Hugging Face repository
    --hf-upload-dir "my_arbitrary_name"
```

## Repository Structure

### Default Repository

By default, results are uploaded to your specified HuggingFace dataset repository.

### Custom Repository

You can specify a different repository with:

```bash
--hf-upload-repo "username/repository-name"
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
