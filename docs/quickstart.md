## Quick Start

The codebase is tested and compatible with Python 3.12 and PyTorch 2.5.
You will also need the appropriate CUDA dependencies and version installed on your system for GPU support.

The easiest way to get started is by installing the library via `pip` and use it as an external dependency.
```
pip install eval_framework
```

There are optional extras available to unlock specific features of the library:
- `mistral` for inference on Mistral models
- `transformers` for inference using the transformers library
- `api` for inference using the aleph-alpha client.
- `vllm` for inference via VLLM
- `determined` for running jobs via determined
- `comet` for the COMET metric

As a short hand, the `all` extra installs all of the above.

For development, you can instead install it directly from the repository instead, please first install
 [uv](https://docs.astral.sh/uv/getting-started/installation/)

To install the project with all optional extras use
```bash
uv sync --all-extras
```

We provide custom groups to control optional extras.
- `flash_attn`: Install `flash_attn` with correct handling of build isolation

Thus, the following will setup the project with `flash_attn`
```bash
uv sync --all-extras --group flash_attn
```

To evaluate a single benchmark locally, you can use the following command:

```bash
eval_framework \
    --models src/eval_framework/llm/models.py \
    --llm-name Smollm135MInstruct \
    --task-name "GSM8K" \
    --output-dir ./eval \
    --num-fewshot 5 \
    --num-samples 10
```

### Your First Evaluation

1. **Install the framework** (see Quick Start above)
```
pip install eval_framework[transformers]
```

2. **Create and run your first evaluation using HuggingFace model**:

   ```python
    from pathlib import Path

    from eval_framework.llm.huggingface import HFLLM
    from eval_framework.main import main
    from eval_framework.tasks.eval_config import EvalConfig
    from template_formatting.formatter import HFFormatter

    # Define your model
    class MyHuggingFaceModel(HFLLM):
        LLM_NAME = "microsoft/DialoGPT-medium"
        DEFAULT_FORMATTER = partial(HFFormatter, "microsoft/DialoGPT-medium")

    if __name__ == "__main__":
        # Initialize your model
        llm = MyHuggingFaceModel()

        # Running evaluation on GSM8K task using 5 few-shot examples and 10 samples
        config = EvalConfig(
            output_dir=Path("./eval_results"),
            num_fewshot=5,
            num_samples=10,
            task_name="GSM8K",
            llm_class=MyHuggingFaceModel,
        )

        # Run evaluation and get results
        results = main(llm=llm, config=config)
   ```

3. **Review results** - Check `./eval_results/` for detailed outputs and use our [results guide](docs/understanding_results_guide.md) to interpret them

### Next Steps

- **Use CLI interface**: See [CLI usage guide](docs/cli_usage.md) for command-line evaluation options
- **Evaluate HuggingFace models**: Follow our [HuggingFace evaluation guide](docs/evaluate_huggingface_model.md)
- **Create custom benchmarks**: Follow our [benchmark creation guide](docs/add_new_benchmark_guide.md)
- **Scale your evaluations**: Use [Determined AI integration](docs/using_determined.md) for distributed evaluation
- **Understand your results**: Read our [results interpretation guide](docs/understanding_results_guide.md)
- **Log results in WandB**: See how [we integrate WandB](docs/wandb_integration.md) for metric and lineage tracking
- For more detailed CLI usage instructions, see the [CLI Usage Guide](docs/cli_usage.md).