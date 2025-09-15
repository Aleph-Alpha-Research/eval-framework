# Aleph Alpha Eval-Framework

> **Comprehensive LLM evaluation at scale** - A production-ready framework for evaluating large language models across 90+ benchmarks.

## Features

- 90+ Benchmarks: Covers reasoning, knowledge, coding, long-context, and safety tasks.
- Custom Benchmarks: Easily add new benchmarks with minimal code using the BaseTask class.
- Distributed Evaluation: Integration with Determined AI for scalable distributed evaluation.
- Docker Support: Pre-configured Dockerfiles for local and distributed setups.
- Flexible Model Integration: Supports models loaded via HuggingFace Transformers or custom implementations using the BaseLLM class.
- Custom Metrics: Easily define new metrics using the BaseMetric class.
- Rich Outputs: Generates JSON results, plots, and detailed analysis reports.
- Perturbation Testing: Robustness analysis with configurable perturbation types and probabilities.
- Statistical Analysis: Includes confidence intervals and significance testing for reliable comparisons.
- LLM-as-a-Judge: Evaluation using LLM judges.

![eval-framework](docs/eval-framework.png "eval-framework")

## Benchmark Coverage & Task Categories

### Core Capabilities

| **Reasoning** | **Knowledge** | **Coding** | **Long Context** |
|---------------|---------------|------------|------------------|
| MMLU (57 subjects) | TriviaQA | HumanEval | InfiniteBench |
| SQuAD v1/v2 | MBPP |
| ARC | Natural Questions | CodeT5 | ZeroSCROLLS |
| HellaSwag | QuAC | Programming | QuALITY |
| Winogrande | COPA | Debugging  |

### Languages & Domains

| **Multilingual** | **Specialized** | **Safety & Bias** | **Efficiency** |
|------------------|-----------------|-------------------|----------------|
| WMT Translation | Legal (CaseHold) | TruthfulQA | Token counting |
| FLORES-200 | Winogender | Latency metrics |
| Multilingual MMLU | Medical (MedQA) | Stereotype detection | Memory usage |
| German/Finnish tasks | Scientific (SciQ) | Harmful content | Cost analysis |

### Completion

Tasks focused on logical reasoning, text distillation, instruction following, and output control. Examples include:
- **AIME 2024:** Logical Reasoning (Math)
- **DUC Abstractive:** Text Distillation (Extraction)
- **Custom Data: Complaint Summarization:** Text Distillation (Summarization)

### Loglikelihoods

Tasks emphasizing classification, reasoning, and open QA. Examples include:
- **Abstract Reasoning Challenge (ARC):** Classification
- **Casehold:** Open QA

### Long-Context

Tasks designed for long-context scenarios, including QA, summarization, and aggregation. Examples include:
- **InfiniteBench_CodeDebug:** Programming
- **ZeroSCROLLS GovReport:** QA (Government)

### Metrics

Evaluation metrics include:
- **Completion Metrics:** Accuracy, Bleu, F1, Rouge
- **Loglikelihood Metrics:** Accuracy Loglikelihood, Probability Mass
- **LLM Metrics:** Chatbot Style Judge, Instruction Judge
- **Efficiency Metrics:** Bytes per Sequence Position

For the full list of tasks and metrics, see [Detailed Task Table](docs/benchmarks_and_metrics.md).

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

There is also a pre-commit hook to help with development:
```
uv run pre-commit install
```

After installation, task documentation can be generated with `uv run python -m eval_framework.utils.generate_task_docs` (see [docs/installation.md(docs/installation.md)) for more details.

## Getting Started

### Understanding the Evaluation Framework

Eval-Framework provides a unified interface for evaluating language models across diverse benchmarks. The framework follows this interaction model:

1. **Define Your Model** - Specify which model to evaluate (HuggingFace, API, or custom)
2. **Choose Your Task** - Select from 150+ available benchmarks or create custom ones
3. **Configure Evaluation** - Set parameters like few-shot examples, sample count, and output format
4. **Run Evaluation** - Execute locally via CLI/script or distribute via Determined AI
5. **Analyze Results** - Review detailed JSON outputs, metrics, and generated reports

### Core Components

- **Models**: Defined via [`BaseLLM`](docs/evaluate_huggingface_model.md) interface (HuggingFace, OpenAI, custom APIs)
- **Tasks**: Inherit from [`BaseTask`](docs/add_new_benchmark_guide.md) (completion, loglikelihood, or LLM-judge based)
- **Metrics**: Automatic scoring via [`BaseMetric`](docs/benchmarks_and_metrics.md) classes
- **Formatters**: Handle prompt construction and model-specific formatting
- **Results**: Structured outputs with sample-level details and aggregated statistics

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

### Example CLI Usage

To evaluate a single benchmark locally, you can use the following command:

```bash
eval_framework \
    --models 'eval_framework.llm.models.Smollm135MInstruct' \
    --task-name "GSM8K" \
    --output-dir ./eval \
    --num-fewshot 5 \
    --num-samples 10
```

For more detailed CLI usage instructions, see the [CLI Usage Guide](docs/cli_usage.md).

## Documentation

### Getting Started

- **[CLI Usage Guide](docs/cli_usage.md)** - Detailed instructions for using the command-line interface
- **[Evaluating HuggingFace Models](docs/evaluate_huggingface_model.md)** - Complete guide for evaluating HuggingFace models
- **[Understanding Results](docs/understanding_results_guide.md)** - How to read and interpret evaluation results

### Advanced Usage

- **[Adding New Benchmarks](docs/add_new_benchmark_guide.md)** - Complete guide with practical examples for adding new benchmarks
- **[Benchmarks and Metrics](docs/benchmarks_and_metrics.md)** - Comprehensive overview of all available benchmarks and evaluation metrics
- **[Overview of Dataloading](docs/overview_dataloading.md)** - Explanation of dataloading and task/sample/message structure

### Scaling & Production

- **[Using Determined](docs/using_determined.md)** - Guide for distributed evaluation using Determined AI
- **[Controlling Upload Results](docs/controlling_upload_results.md)** - How to manage and control the upload of evaluation results

### Citation

If you use `eval-framework` in your research, please cite:

```bibtex
@software{eval_framework,
  title={Aleph Alpha Eval Framework},
  year={2025},
  url={https://github.com/Aleph-Alpha-Research/eval-framework}
}
```

### License

This project is licensed under the [Apache License 2.0](LICENSE).

<br><br>
---

This project has received funding from the European Union’s Digital Europe Programme under grant agreement No. 101195233 (OpenEuroLLM).

The contents of this publication are the sole responsibility of the OpenEuroLLM consortium and do not necessarily reflect the opinion of the European Union.

<p align="center">
  <img src="docs/OELLM_1.png" alt="OELLM 1" width="100" style="margin-right: 50px;"/>
  <img src="docs/OELLM_2.png" alt="OELLM 2" width="350"/>
</p>
