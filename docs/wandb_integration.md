# Weights and Biases Integration with Eval-Framework

## Overview
The evaluation framework supports logging results to Weights and Biases (WandB) and loading registered model checkpoints.

## Benefits and Enablement

- **Centralized eval tracking**: Automatically log evaluation metrics
- **Centralized checkpoint Storage**: Discover and reference checkpoints from a central location
- **Collaboration**: Share results and models with team members through WandB's web interface

## Registered Models

The eval-framework can load models from your WandB Model Registry.

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

### Environment Variables

**Required:**
- `WANDB_API_KEY`: Your WandB API key from [wandb.ai/authorize](https://wandb.ai/authorize)

**Optional (for S3-backed artifacts):**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_ENDPOINT_URL`

*Note: AWS variables are only needed if using reference-backed artifacts. Direct WandB artifact storage doesn't require them.*
