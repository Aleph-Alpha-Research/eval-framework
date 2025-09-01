# Installation and Dependencies

This guide provides detailed installation instructions and dependency information for the eval-framework.

## Installation Method
#### Install uv

Follow the official [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

#### Install Eval Framework

```bash
# Clone the repository
git clone https://github.com/Aleph-Alpha-Research/eval-framework/tree/main
cd eval-framework

# Install all dependencies including optional extras
uv sync --all-extras

# Install flash-attention with CUDA 12.4 (requires compilation)
uv sync --all-extras --group cu124 --group flash-attn
```

#### Generate task documentation

Generate task documentation

Extra documentation can be automatically generated for all tasks available in `eval-framework` as well as for a
specified set of extra task modules with the utility script `utils/generate-task-docs.py`. The script supports a few
command line arguments:
- `--only-tasks`: a list of task names to generate documentation for. If empty, all tasks will be processed.
- `--exclude-tasks`: a list of task names to exclude from documentation generation.
- `--extra-task-modules`: a list of files and folders containing additional task definitions.
- `--add-prompt-examples`: if set, examples prompts for each of the formatters will be added in the generated docs.
- `--formatter`: specify which formatter to use for formatting the task samples. If not explicitly specified, default
formatters will be used.

The generated documentation will be saved in the `docs/tasks` directory, with each task having its own markdown file.
A [README.md](tasks/README.md) file will also be generated in the `docs/tasks` directory, listing all the tasks and
linking to their documentation.

Run with:
```
uv run python utils/generate-task-docs.py
```

By default, formatted prompt examples are not included to this documentation file. Those can be added is you run instead:
```
uv run python utils/generate-task-docs.py --add-prompt-examples
```

## Environment Configuration

### Environment Variables

Create a `.env` file in the project root. Essential environment variables:

```bash
# API Keys (if using external models)
HF_TOKEN="your_huggingface_token"          # For private HF models
OPENAI_API_KEY="your_openai_key"           # For OpenAI models as judges
AA_TOKEN="your_aleph_alpha_token"          # For Aleph Alpha API

# Optional: Inference endpoints
AA_INFERENCE_ENDPOINT="your_inference_url"

# Debug mode
DEBUG=false


## Docker Installation

For containerized deployment:

### Available Dockerfiles

The repository contains multiple Dockerfiles for different use cases:

1. **`Dockerfile`** (Main) - General evaluation framework with CUDA support
2. **`Dockerfile_codebench`** - Specialized for BigCodeBench coding tasks
3. **`Dockerfile_Determined`** - For Determined.ai cluster deployments

### Build from Source

#### Main Evaluation Framework

```bash
# Build main image (uses Dockerfile)
# This creates a CUDA-enabled container with Python 3.12, uv, and all framework dependencies
docker build -t eval-framework .

# Run with GPU support
docker run -it --gpus all -v $(pwd):/workspace eval-framework
```


#### Specialized Builds

```bash
# For coding evaluation tasks (BigCodeBench)
docker build -f Dockerfile_codebench -t eval-framework-codebench .

# For Determined.ai cluster deployment (requires base image)
docker build -f Dockerfile_Determined -t eval-framework-determined .
```
