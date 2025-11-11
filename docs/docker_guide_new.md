# Docker Guide

Guide for using eval-framework with Docker for both AA users and external contributors.


### Build from Repository

**Latest Release:**

```bash
git clone https://github.com/Aleph-Alpha-Research/eval-framework.git
cd eval-framework
docker build -t eval_framework:latest .
```

**Specific Version:**

```bash
git clone https://github.com/Aleph-Alpha-Research/eval-framework.git
cd eval-framework
git checkout v0.2.3
docker build -t eval_framework:v0.2.3 .
```



### Build from PyPI

Create `Dockerfile.pypi`:

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir eval_framework[all]==0.2.3

WORKDIR /workspace
ENTRYPOINT ["eval_framework"]
```

Build:

```bash
docker build -t eval_framework:v0.2.3 -f Dockerfile.pypi .
```

### For Aleph Alpha Users

AA users have access to pre-built Docker images via GitLab registry.

```bash
# Authenticate
docker login registry.gitlab.aleph-alpha.de

# Pull specific version
docker pull registry.gitlab.aleph-alpha.de/research/public-registry/eval_framework:v0.2.3

# Run evaluation
docker run --gpus all \
  -v $(pwd)/results:/results \
  registry.gitlab.aleph-alpha.de/research/public-registry/eval_framework:v0.2.3 \
  eval_framework --task-name MMLU --output-dir /results
```


#### Available Tags

Check [PyPI releases](https://pypi.org/project/eval-framework/) or [main README](../README.md) for current versions.

```bash
# Specific version
docker pull registry.gitlab.aleph-alpha.de/research/public-registry/eval_framework:v0.2.3

# Minor version
docker pull registry.gitlab.aleph-alpha.de/research/public-registry/eval_framework:v0.2

# Latest stable
docker pull registry.gitlab.aleph-alpha.de/research/public-registry/eval_framework:latest

# Specific branch
docker pull registry.gitlab.aleph-alpha.de/research/public-registry/eval_framework:feature-branch-name
```




## Prerequisites and Configuration

### Required API Keys

| Key | Required For |
|-----|-------------|
| `HF_TOKEN` | HuggingFace model downloads, gated datasets |
| `WANDB_API_KEY` | Experiment tracking (optional) |
| `GL_REGISTRY_TOKEN` | Pulling AA Docker images |

**When are these needed?**

- `HF_TOKEN`: Only required if you're downloading gated models (e.g., Llama, Mistral) or private datasets. Public models work without it.
- `WANDB_API_KEY`: Optional. Only needed if you want to log experiments to Weights & Biases.
- `GL_REGISTRY_TOKEN`: Only for AA users pulling pre-built Docker images from GitLab registry.

**Note:** For basic evaluations with public models, you can skip this section entirely.

### Setting Up Keys

**Export in shell:**

```bash
export HF_TOKEN="hf_your_token_here"
export WANDB_API_KEY="your_wandb_key_here"
```

**Pass at runtime:**

```bash
docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  eval_framework:v0.2.3 [command]
```

**Using .env file:**

Create `.env`:

```bash
HF_TOKEN=hf_your_token_here
WANDB_API_KEY=your_wandb_key_here
```

Run with env file:

```bash
docker run --env-file .env eval_framework:v0.2.3 [command]
```


## Running Evaluations



### GPU Configuration

```bash
# All GPUs
docker run --gpus all eval_framework:v0.2.3 [command]

# Specific GPU
docker run --gpus 0 eval_framework:v0.2.3 [command]

# Multiple GPUs
docker run --gpus '"device=0,1"' eval_framework:v0.2.3 [command]
```


### Interactive Shell

```bash
docker run -it --gpus all \
  -v $(pwd):/workspace \
  eval_framework:v0.2.3 \
  /bin/bash
```

## Determined AI Integration

Add to `experiment.yaml`:

```yaml
name: eval-framework-experiment

environment:
  image: registry.gitlab.aleph-alpha.de/research/public-registry/eval_framework:v0.2.3
  registry_auth:
    username: token
    password: $GL_REGISTRY_TOKEN
  environment_variables:
    - HF_TOKEN
    - WANDB_API_KEY

resources:
  slots_per_trial: 1

entrypoint: |
  eval_framework \
    --task-name MMLU \
    --output-dir /tmp/results \
    --num-fewshot 5
```

Submit:

```bash
export HF_TOKEN="hf_your_token_here"
export GL_REGISTRY_TOKEN="glpat_your_token_here"
det experiment create experiment.yaml .
```

## Example Workflows

### Single Evaluation

```bash
docker run --gpus all \
  -v $(pwd)/results:/results \
  eval_framework:v0.2.3 \
  eval_framework \
    --models /eval_framework/src/eval_framework/llm/models.py \
    --llm-name Smollm135MInstruct \
    --task-name MMLU \
    --task-subjects abstract_algebra \
    --output-dir /results \
    --num-fewshot 5 \
    --num-samples 10
```

### Batch Evaluations

```bash
#!/bin/bash
for task in MMLU HellaSwag ARC TriviaQA; do
  docker run --gpus all \
    -v $(pwd)/results:/results \
    eval_framework:v0.2.3 \
    eval_framework \
      --task-name $task \
      --output-dir /results/$task \
      --num-fewshot 5
done
```
