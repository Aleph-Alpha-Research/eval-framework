# local-vllm-server

Small wrapper that spawns a local `vllm serve` subprocess and exposes it through the eval-framework `BaseLLM` interface (via `OpenAIModel` pointed at the local endpoint).

This lives outside the main `eval-framework` package because vLLM's dependency tree (torch, CUDA, transformers pins) frequently conflicts with the rest of the evaluation stack. Install it into its own environment when you need it.

## Install

Requires a working `vllm` runtime (GPU, drivers, etc.). From this directory:

```bash
uv sync
```

`eval-framework` is pulled in as an editable path dependency (see `[tool.uv.sources]` in `pyproject.toml`), so the version of the framework you get is whatever is currently checked out in the parent directory.

## Programmatic usage

```python
from local_vllm_server import VLLMLocalServerModel

llm = VLLMLocalServerModel(
    model_name="Qwen/Qwen3-0.6B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.3,
    dtype="float16",
    max_model_len=256,
    enforce_eager=True,
)
```

## Running a full eval via the eval-framework CLI

The `eval-framework` package installs the `eval_framework` CLI. Because `VLLMLocalServerModel` is a normal `BaseLLM`, you can pass its dotted path to `--llm-name` and feed constructor arguments through `--llm-args`. Run this from inside the `local-vllm-server/` directory (so `uv run` picks up this environment, with vLLM available):

```bash
uv run eval_framework \
    --llm-name local_vllm_server.VLLMLocalServerModel \
    --llm-args \
        model_name=Qwen/Qwen3-0.6B \
        tensor_parallel_size=1 \
        gpu_memory_utilization=0.3 \
        dtype=float16 \
        max_model_len=2048 \
        enforce_eager=True \
    --task-name MMLU \
    --task-subjects abstract_algebra \
    --output-dir ./eval_results \
    --num-fewshot 5 \
    --num-samples 10
```

The wrapper starts `vllm serve` in a subprocess, blocks until the OpenAI-compatible endpoint is up, then routes all completions through it. Cleanup happens automatically at process exit.

## Tests

```bash
uv run pytest
```

Tests are marked `gpu` and require a real GPU with vLLM working end to end.
