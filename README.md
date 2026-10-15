<!-- Badges -->
<div align="center">

# Aleph Alpha Eval-Framework

**Comprehensive LLM evaluation at scale** - A production-ready framework for evaluating large language models across 90+ benchmarks.

[![Build Status](https://github.com/Aleph-Alpha-Research/eval-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/Aleph-Alpha-Research/eval-framework/actions)
[![Version](https://img.shields.io/github/v/release/Aleph-Alpha-Research/eval-framework)](https://github.com/Aleph-Alpha-Research/eval-framework/releases)
[![PyPI](https://img.shields.io/pypi/v/eval-framework.svg)](https://pypi.org/project/eval-framework/)
[![License](https://img.shields.io/github/license/Aleph-Alpha-Research/eval-framework.svg)](LICENSE)

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://aleph-alpha-research.github.io/eval-framework/)
[![Stars](https://img.shields.io/github/stars/Aleph-Alpha-Research/eval-framework)](https://github.com/Aleph-Alpha-Research/eval-framework/stargazers)

![eval-framework](https://raw.githubusercontent.com/Aleph-Alpha-Research/eval-framework/refs/heads/main/docs/eval-framework.png)

</div>

## Why Choose This Framework?

- **Scalability**: Built for distributed evaluation. Currently providing an integration with Determined AI.
- **Extensibility**: Easily add custom models, benchmarks, and metrics with object-oriented base classes.
- **Comprehensive**: Comes pre-loaded with over 90 tasks covering a broad and diverse range, from reasoning and coding to safety and long-context. Also comes with a comprehensive set of metrics, including LLM-as-a-judge evaluations.

## Other features

- Flexible Model Integration: Supports OpenAI-compatible endpoints or custom implementations using the BaseLLM class.
- Custom Benchmarks: Easily add new benchmarks with minimal code using the BaseTask class.
- Custom Metrics: Easily define new metrics using the BaseMetric class.
- Perturbation Testing: Robustness analysis with configurable perturbation types and probabilities.
- Rich Outputs: Generates JSON results, plots, and detailed analysis reports.
- Statistical Analysis: Includes confidence intervals and significance testing for reliable comparisons.
- Docker Support: Pre-configured Dockerfiles for local and distributed setups.

For full documentation, visit our [Docs Page](https://aleph-alpha-research.github.io/eval-framework/).

## Quick Start

The codebase is tested and compatible with Python 3.12. Detailed installation instructions can be found [here](https://aleph-alpha-research.github.io/eval-framework/installation.html).

The easiest way to get started is by installing the library via `pip` and use it as an external dependency.
```
pip install eval_framework
```

There are optional extras available to unlock specific features of the library:
- `determined` for running jobs via determined.

As a short hand, the `all` extra installs all of the above.

We use `uv` to better resolve dependencies when downloading the extras. You can install uv with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or by follwing the `uv` [installation docs.](https://docs.astral.sh/uv/getting-started/installation/)

Now, you can safely install the project with all optional extras:
```bash
uv sync --all-extras
```
or with pip
```bash
uv pip install eval_framework[all]
```

Tip: ensure python is properly installed with uv:
```
uv python install 3.12 --reinstall
```

To evaluate a single benchmark locally, you can use the following command. The example below talks to the OpenAI API and needs `OPENAI_API_KEY` to be set. Any OpenAI-compatible server (for example vLLM) can be targeted by adding `base_url=http://host:port/v1` to `--llm-args`.

```bash
eval_framework \
    --llm-name eval_framework.llm.openai.OpenAIModel \
    --llm-args model_name="gpt-4o-mini" \
    --task-name "MMLU" \
    --task-subjects "abstract_algebra" \
    --output-dir ./eval_results \
    --num-fewshot 5 \
    --num-samples 10
```

For more detailed CLI usage instructions, see the [CLI Usage Guide](https://aleph-alpha-research.github.io/eval-framework/cli_usage.html).

## Benchmark Coverage & Task Categories

### Core Capabilities

Subset of core capabilities benchmarks coverd by `eval-framework`:

| **Reasoning** | **Knowledge** | **Math** | **Coding** | **Structured outputs** | **Long Context** |
|---------------|---------------|----------|------------|------------------------|------------------|
| COPA, BalancedCOPA | ARC | AIME | BigCodeBench | IFEval | InfiniteBench |
| Hellaswag | MMLU | GSM8K | HumanEval | StructEval | QUALITY |
| Winogrande | Openbook QA| MATH-500 | MBPP | | ZeroSCROLLS |


### Languages & Domains

Subset of language-specific and domain-specific benchmarks coverd by `eval-framework`:

| **Multilingual** | **Specialized** | **Safety & Bias** | **Efficiency Metrics** |
|------------------|-----------------|-------------------|----------------|
| WMT Translation | MMLU | TruthfulQA | Compression ratios |
| FLORES-200 | Legal (CaseHold) | Winogender | Runtime |
| Multilingual MMLU | Scientific (SciQ) | | |
| German/Finnish tasks |  | | |

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

For the full list of tasks and metrics, see [Detailed Task Table](https://aleph-alpha-research.github.io/eval-framework/benchmarks_and_metrics.html).

## Getting Started

### Understanding the Evaluation Framework

Eval-Framework provides a unified interface for evaluating language models across diverse benchmarks. The framework follows this interaction model:

1. **Define Your Model** - Specify which model to evaluate (OpenAI-compatible endpoint or custom)
2. **Choose Your Task** - Select from 150+ available benchmarks or create custom ones
3. **Configure Evaluation** - Set parameters like few-shot examples, sample count, and output format
4. **Run Evaluation** - Execute locally via CLI/script or distribute via Determined AI
5. **Analyze Results** - Review detailed JSON outputs, metrics, and generated reports

### Core Components

- **Models**: Defined via the `BaseLLM` interface (OpenAI-compatible endpoints, custom APIs); see the [Model Arguments guide](https://aleph-alpha-research.github.io/eval-framework/model_arguments.html)
- **Tasks**: Inherit from [`BaseTask`](https://aleph-alpha-research.github.io/eval-framework/add_new_benchmark_guide.html) (completion, loglikelihood, or LLM-judge based)
- **Metrics**: Automatic scoring via [`BaseMetric`](https://aleph-alpha-research.github.io/eval-framework/benchmarks_and_metrics.html) classes
- **Formatters**: Handle prompt construction and model-specific formatting
- **Results**: Structured outputs with sample-level details and aggregated statistics

### Your First Evaluation

1. **Install the framework** (see Quick Start above)
```
pip install eval_framework
```

2. **Create and run your first evaluation against an OpenAI-compatible endpoint** (requires `OPENAI_API_KEY`; pass `base_url=...` to the constructor to target a self-hosted server such as vLLM):

```python
from pathlib import Path

from eval_framework.llm.openai import OpenAIModel
from eval_framework.main import main
from eval_framework.tasks.eval_config import EvalConfig


# Define your model
class MyModel(OpenAIModel):
    LLM_NAME = "gpt-4o-mini"


if __name__ == "__main__":
    # Initialize your model
    llm = MyModel()

    # Running evaluation on MMLU abstract algebra task using 5 few-shot examples and 10 samples
    config = EvalConfig(
        output_dir=Path("./eval_results"),
        num_fewshot=5,
        num_samples=10,
        task_name="MMLU",
        task_subjects=["abstract_algebra", "astronomy"],
        llm_class=MyModel,
    )

    # Run evaluation and get results
    results = main(llm=llm, config=config)
```

3. **Review results** - Check `./eval_results/` for detailed outputs and use our [results guide](https://aleph-alpha-research.github.io/eval-framework/understanding_results_guide.html) to interpret them

### Next Steps

- **Use CLI interface**: See [CLI usage guide](https://aleph-alpha-research.github.io/eval-framework/cli_usage.html) for command-line evaluation options
- **Understand model arguments**: Read out [Model Arguments guide](https://aleph-alpha-research.github.io/eval-framework/model_arguments.html)
- **Create custom benchmarks**: Follow our [benchmark creation guide](https://aleph-alpha-research.github.io/eval-framework/add_new_benchmark_guide.html)
- **Scale your evaluations**: Use [Determined AI integration](https://aleph-alpha-research.github.io/eval-framework/using_determined.html) for distributed evaluation
- **Understand your results**: Read our [results interpretation guide](https://aleph-alpha-research.github.io/eval-framework/understanding_results_guide.html)
- **Log results in WandB**: See how [we integrate WandB](https://aleph-alpha-research.github.io/eval-framework/wandb_integration.html) for metric and lineage tracking

## Documentation

### Getting Started

- **[CLI Usage Guide](https://aleph-alpha-research.github.io/eval-framework/cli_usage.html)** - Detailed instructions for using the command-line interface
- **[Understanding Results](https://aleph-alpha-research.github.io/eval-framework/understanding_results_guide.html)** - How to read and interpret evaluation results

### Advanced Usage

- **[Understanding Model Arguments](https://aleph-alpha-research.github.io/eval-framework/model_arguments.html)** - Thorough guide on each constructor argument for salient model classes
- **[Adding New Benchmarks](https://aleph-alpha-research.github.io/eval-framework/add_new_benchmark_guide.html)** - Complete guide with practical examples for adding new benchmarks
- **[Benchmarks and Metrics](https://aleph-alpha-research.github.io/eval-framework/benchmarks_and_metrics.html)** - Comprehensive overview of all available benchmarks and evaluation metrics
- **[Overview of Dataloading](https://aleph-alpha-research.github.io/eval-framework/overview_dataloading.html)** - Explanation of dataloading and task/sample/message structure

### Scaling & Production

- **[Using Determined](https://aleph-alpha-research.github.io/eval-framework/using_determined.html)** - Guide for distributed evaluation using Determined AI
- **[Controlling Upload Results](https://aleph-alpha-research.github.io/eval-framework/controlling_upload_results.html)** - How to manage and control the upload of evaluation results

### Contributing

- **[Contributing Guide](https://aleph-alpha-research.github.io/eval-framework/CONTRIBUTING.html)** - Guide for contributing to this project
- **[Testing](https://aleph-alpha-research.github.io/eval-framework/testing.html)** - Guide for running tests comparable to the CI pipelines

### Citation

If you use `eval-framework` in your research, please cite:

```bibtex
@software{eval_framework,
  author={Aleph Alpha Research},
  title={Aleph Alpha Eval Framework},
  year={2026},
  version = {x.y.z},
  url={https://github.com/Aleph-Alpha-Research/eval-framework}
}
```

### License

This project is licensed under the [Apache License 2.0](LICENSE).

The constituent tasks' datasets can be subject to specific, more restrictive license terms by third parties. By using `eval-framework` you agree to adhere to and be bound by such license terms in the individual case.

<br><br>
---

This project has received funding from the European Union’s Digital Europe Programme under grant agreement No. 101195233 (OpenEuroLLM).

The contents of this publication are the sole responsibility of the OpenEuroLLM consortium and do not necessarily reflect the opinion of the European Union.

<p align="center">
  <img src="https://raw.githubusercontent.com/Aleph-Alpha-Research/eval-framework/main/docs/OELLM_1.png" width="100" style="margin-right: 50px;"/>
  <img src="https://raw.githubusercontent.com/Aleph-Alpha-Research/eval-framework/main/docs/OELLM_2.png" width="350"/>
</p>
