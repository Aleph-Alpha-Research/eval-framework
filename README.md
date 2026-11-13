# Aleph Alpha Eval-Framework

[![Build Status](https://github.com/Aleph-Alpha-Research/eval-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/Aleph-Alpha-Research/eval-framework/actions)
[![Version](https://img.shields.io/github/v/release/Aleph-Alpha-Research/eval-framework)](https://github.com/Aleph-Alpha-Research/eval-framework/releases)
[![PyPI](https://img.shields.io/pypi/v/eval-framework.svg)](https://pypi.org/project/eval-framework/)
[![License](https://img.shields.io/github/license/Aleph-Alpha-Research/eval-framework.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://aleph-alpha-research.github.io/eval-framework/)

A suite of curated benchmarks for evaluating Large Language Models (LLMs).

## Key Features

- ✅ **Validated:** Benchmarks implemented in this suite are required to have reproduced published results.
- ⚖️ **Legal:** The data is legal to exist and to use. Some datasets are subject to third-party licenses.
- 📐 **Metrics:** Bayesian loglikelihood accuracy with a length correction fitted from the data
  ([paper](https://arxiv.org/abs/2607.12767)), BitsPerByte on answers with corpus- and prior-level variants, and the
  distributional-correctness score ([Burns, 2025](https://openreview.net/forum?id=fiMZJJzZ7g)).
- 🤷 **Abstention variants:** `ARC_IDK`, `MMLU_IDK`, `MMLU_PRO_IDK`, `PIQA_IDK` and `GoldenSwag_IDK` add a scored
  "I don't know" answer, so a model can decline instead of guessing.
- 🌍 **European languages:** EllaMind translations of ARC, CommonsenseQA, GPQA, GSM8K, HellaSwag, PIQA and more, in
  German, French, Spanish and Italian.
- 📚 **250+ tasks:** including BigCodeBench, GoldenSwag, LBPP and MultiPL-E. Browse the
  [benchmarks source directory](https://github.com/Aleph-Alpha-Research/eval-framework/tree/main/src/eval_framework/benchmarks).

## Quick start

The quickstart runs a small model on your machine with [llama.cpp](https://github.com/ggml-org/llama.cpp). No API key is needed.
It requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/Aleph-Alpha-Research/eval-framework
cd eval-framework
uv sync
brew install llama.cpp   # other platforms: see the llama.cpp install guide
```

Then run a benchmark:

```bash
uv run eval_framework \
    --llm-args model_name=Qwen/Qwen3-0.6B-GGUF \
    --task-name MMLU_COT \
    --task-subjects abstract_algebra \
    --num-samples 10 \
    --max-tokens 1024 \
    --output-dir ./eval_results
```

**What happens here?**

- The CLI starts the model with llama.cpp and stops it when the run ends.
- It downloads the MMLU dataset and asks the model 10 abstract-algebra questions (the first run also downloads
the model).
- It writes the scores to `aggregated_results.json` and every prompt and answer to `output.jsonl`, in a run
folder under `./eval_results`.

**Example output** (from `aggregated_results.json`, shortened):

```json
{
    "Average Accuracy Completion": 0.2,
    "ErrorFreeRatio Accuracy Completion": 1.0
}
```

Scores are fractions between 0 and 1. This 0.6B model is only a smoke test that the pipeline works: it often
runs out of tokens or misses the required answer format, so expect a low score.

## Other models

Hosted or self-hosted, any OpenAI-compatible endpoint works:

```bash
# OpenAI
export OPENAI_API_KEY=...
uv run eval_framework --llm-args model_name=gpt-4o-mini \
    --task-name MMLU_COT --task-subjects abstract_algebra --num-samples 10

# Your own server, e.g. vLLM
uv run eval_framework --llm-args model_name=<served model> base_url=http://localhost:8000/v1 \
    --task-name MMLU_COT --task-subjects abstract_algebra --num-samples 10
```

## Python API

eval-framework can also be called from your own Python code instead of the command line. This is needed for the
loglikelihood metrics and for custom model classes. See
[Python usage](https://aleph-alpha-research.github.io/eval-framework/python_usage.html).

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
