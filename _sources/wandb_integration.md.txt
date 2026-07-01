# Weights and Biases Integration with Eval-Framework

## Overview
The evaluation framework supports logging results to Weights and Biases (WandB) and loading registered model checkpoints.

## Benefits and Enablement

- **Centralized eval tracking**: Automatically log evaluation metrics
- **Centralized checkpoint Storage**: Discover and reference checkpoints from a central location
- **Collaboration**: Share results and models with team members through WandB's web interface

## Registered Models and Results

The eval-framework can load models from your WandB Model Registry and can upload results as WandB artifacts.

This enables:
- **Version control**: Track model checkpoints with versioning and aliases
- **Metadata management**: Store model descriptions and additional metadata per version
- **Centralized discovery**: Browse and search models
- **Lineage tracking**: Maintain audit trails

### Storage Location

We currently support one of two storage backends:
- **WandB Cloud**: Default storage in WandB's managed infrastructure
- **S3-backed artifacts**: S3-compatible buckets (see the [Environment Variables](#environment-variables) section for AWS configuration)

## Evaluation Run Logging

This integration automatically:
- **Groups runs by checkpoint name**: Organizes evaluation results by model checkpoint
- **Logs evaluation metrics and configuration**: Log metrics and run settings
- **Records eval-framework version**: Tracks the eval-framework version used during runs
- **References HuggingFace upload paths**: Provides a link to full result upload locations when available
- **Maintains model lineage**: Links eval runs to a particular model and model version

**Experiment details**:

Runs are grouped within projects by checkpoint name and version. Additional hierarchical groupings are available, but not limited to the following:

- Language
- Benchmark Task
- Fewshot
- Number of samples
- Metric

## Usage
WandB logging is disabled by default. To enable it, set up a valid WandB account and configure the required environment variables in your `.env` file:

```
# Weights & Biases configuration
WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"
```

### Method 1: CLI Upload

Add the `--wandb-project` (and potentially `--wandb-entity` if not the default one) to your CLI command:

```bash
uv run eval_framework \
    --context local \
    --models tests/conftest.py \
    --llm-name Llama31_8B_Instruct_HF \
    --task-name ARC \
    --num-fewshot 3 \
    --num-samples 100 \
    --output-dir "./test_outputs_folder" \
    --wandb-project "my_wandb_project"
```

### Method 2: Determined Configuration

Add `wandb_project` (and potentially `wandb_entity` if not the default one) as a hyperparameter in your Determined experiment config:

```yaml
hyperparameters:
  experiment_name: "my_experiment"
  llm_name: "Llama31_8B_Instruct_HF"
  wandb_project: "my_wandb_project"
  task_args:
    - task_name: "ARC"
      num_fewshot: 3
      num_samples: 100
```

### Environment Variables

**Required:**
- `WANDB_API_KEY`: Your WandB API key from [wandb.ai/authorize](https://wandb.ai/authorize)

**Optional (for S3-backed artifacts):**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_ENDPOINT_URL`

*Note: AWS variables are only needed if using reference-backed artifacts. Direct WandB artifact storage doesn't require them.*

**Custom ones:**
- `WANDB_CACHE_SKIP`: Whether to use W&B cache when downloading model artifacts (defaults to False to avoid double storage).
- `WANDB_ARTIFACT_DIR`: Directory where model artifacts will be downloaded (if not given, a temporary one will be used).
- `WANDB_ARTIFACT_WAIT_TIMEOUT_SEC`: How long to wait for an artifact to become available on W&B if a corresponding "-local" version of the artifact is available.
