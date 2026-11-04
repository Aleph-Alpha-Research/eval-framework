# Aleph Alpha Eval-Framework

[![Build Status](https://github.com/Aleph-Alpha-Research/eval-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/Aleph-Alpha-Research/eval-framework/actions)
[![Version](https://img.shields.io/github/v/release/Aleph-Alpha-Research/eval-framework)](https://github.com/Aleph-Alpha-Research/eval-framework/releases)
[![PyPI](https://img.shields.io/pypi/v/eval-framework.svg)](https://pypi.org/project/eval-framework/)
[![License](https://img.shields.io/github/license/Aleph-Alpha-Research/eval-framework.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://aleph-alpha-research.github.io/eval-framework/)

A suite of curated benchmarks for evaluating **L**arge **L**anguage **M**odels (LLMs)

Benchmarks in this suite are:

* Validated: Benchmarks implemented in this suite are required to have reproduced published results
* Legal: The data is legal to exist and to use. Some datasets are subject to third party licenses
* Comprehensive: For a list of supported benchmarks execute the CLI tool or check the benchmarks source directory. <https://github.com/Aleph-Alpha-Research/eval-framework/tree/main/src/eval_framework/benchmarks>

For full documentation, visit our [Docs Page](https://aleph-alpha-research.github.io/eval-framework/).

## Usage

### CLI

This command evaluates a single benchmark locally. Needs `OPENAI_API_KEY` to be set. Other inference is also supported via `--llm-args`.

```shell
eval_framework \
    --llm-args model_name="gpt-4o-mini" \
    --task-name "MMLU" \
    --task-subjects "abstract_algebra" \
    --output-dir ./eval_results \
    --num-fewshot 5 \
    --num-samples 10
```

For more detailed CLI usage instructions, see the [CLI Usage Guide](https://aleph-alpha-research.github.io/eval-framework/cli_usage.html).

### Python

```Python

from pathlib import Path

from eval_framework.llm.openai import OpenAIModel
from eval_framework.main import main
from eval_framework.tasks.eval_config import EvalConfig

config = EvalConfig(
    llm_class=OpenAIModel,
    llm_args={"model_name": "gpt-4o-mini"},  # plus e.g. "base_url" for any OpenAI-compatible endpoint
    task_name="MMLU",
    task_subjects=["abstract_algebra"],
    num_fewshot=5,
    num_samples=10,  # per subject; None evaluates the whole split
    output_dir=Path("outputs"),
)

results = main(llm=config.llm_class(**config.llm_args), config=config)
```

## Installation

Both the eval-framework library as well as the CLI are distributed via pypi.org as Python Wheels. Thus they can be installed using `pip`. Current minimum supported Python version is Python 3.12.

```shell
pip install eval-framework
```

## Citation

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

## License

This project is licensed under the [Apache License 2.0](LICENSE).

The constituent tasks' datasets can be subject to specific, more restrictive license terms by third parties. By using `eval-framework` you agree to adhere to and be bound by such license terms in the individual case.

---

This project has received funding from the European Union’s Digital Europe Programme under grant agreement No. 101195233 (OpenEuroLLM).

The contents of this publication are the sole responsibility of the OpenEuroLLM consortium and do not necessarily reflect the opinion of the European Union.

<p align="center">
  <img src="https://raw.githubusercontent.com/Aleph-Alpha-Research/eval-framework/main/docs/OELLM_1.png" width="100" style="margin-right: 50px;"/>
  <img src="https://raw.githubusercontent.com/Aleph-Alpha-Research/eval-framework/main/docs/OELLM_2.png" width="350"/>
</p>
