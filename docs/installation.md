# Installation

This guide provides detailed installation instructions and dependency information for the **eval-framework**.

### 1. Install `uv`

Follow the official [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Install Eval Framework

Clone the repository and install all dependencies, including optional extras:

```bash
# Clone the repository
git clone https://github.com/Aleph-Alpha-Research/eval-framework/tree/main
cd eval-framework

# Install all dependencies
uv sync --all-extras

# Install flash-attention optional extra (requires compilation)
uv sync --all-extras --group flash-attn
```

### 3. Test Your Installation

```bash
uv run eval_framework \
    --models src/eval_framework/llm/models.py \
    --llm-name Smollm135MInstruct \
    --task-name "MMLU" \
    --task-subjects "abstract_algebra" "anatomy" \
    --output-dir ./eval_results \
    --num-fewshot 5 \
    --num-samples 10
```

<!-- #### Pre-commit Hooks

To help with development, enable pre-commit hooks:

```bash
uv tool install pre-commit
uv run pre-commit install
```

### 3. Generate Task Documentation

The framework provides a utility script to generate task documentation automatically.

```bash
uv run python -m eval_framework.utils.generate_task_docs
``` -->
<!--
#### Command-Line Options

| Option                  | Description                                                                       |
| ----------------------- | --------------------------------------------------------------------------------- |
| `--only-tasks`          | Comma-separated list of task names to include. If empty, all tasks are processed. |
| `--exclude-tasks`       | List of task names to exclude.                                                    |
| `--extra-task-modules`  | Additional files or folders containing task definitions.                          |
| `--add-prompt-examples` | Include example prompts for each formatter.                                       |
| `--formatter`           | Specify a formatter to use for task samples. Defaults to framework defaults.      |

> The generated documentation is saved in the `docs/tasks` directory. Each task will have its own markdown file.
> A `README.md` file listing all tasks and linking to their documentation is also generated.

Example with formatted prompt examples:

```bash
uv run python -m eval_framework.utils.generate_task_docs --add-prompt-examples
``` -->

## Environment Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (if using external models)
HF_TOKEN="your_huggingface_token"        # For private HuggingFace models
OPENAI_API_KEY="your_openai_key"         # For OpenAI models as judges
AA_TOKEN="your_aleph_alpha_token"        # For Aleph Alpha API

# Optional: Inference endpoints
AA_INFERENCE_ENDPOINT="your_inference_url"

# Debug mode
DEBUG=false
```

## Docker Installation

### Available Dockerfiles

| Dockerfile              | Purpose                                     |
| ----------------------- | ------------------------------------------- |
| `Dockerfile`            | Main evaluation framework with CUDA support |
| `Dockerfile_codebench`  | Specialized for BigCodeBench coding tasks   |
| `Dockerfile_Determined` | For Determined.ai cluster deployments       |


### Build from Source

#### Main Evaluation Framework

```bash
# Build main image (uses Dockerfile)
docker build -t eval-framework .

# Run with GPU support
docker run -it --gpus all -v $(pwd):/workspace eval-framework
```

#### Specialized Builds

```bash
# BigCodeBench coding tasks
docker build -f Dockerfile_codebench -t eval-framework-codebench .

# Determined.ai cluster deployment
docker build -f Dockerfile_Determined -t eval-framework-determined .
```
