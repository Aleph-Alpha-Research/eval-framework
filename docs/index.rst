Eval-Framework Documentation
============================

Welcome to **Aleph Alpha Eval-Framework v0.2.5**.

A **production-ready framework for evaluating large language models** across 90+ benchmarks, covering reasoning, coding, safety, and long-context tasks. The framework provides flexible model integration, custom benchmarks and metrics, perturbation testing, rich outputs, and statistical analysis. It supports local and distributed evaluations, including Determined AI integration.

.. image:: eval-framework.png
   :alt: Eval-Framework
   :align: center

Key Features
------------

- **Scalability:** Distributed evaluation with Determined AI support.
- **Extensibility:** Easily add custom models, benchmarks, and metrics.
- **Comprehensive Benchmarks:** 90+ tasks covering reasoning, coding, math, knowledge, long-context, and safety.
- **Flexible Model Integration:** HuggingFace, custom APIs, and BaseLLM-based models.
- **Robust Metrics:** Completion metrics, loglikelihood metrics, LLM-as-a-judge evaluations, and efficiency metrics.
- **Perturbation Testing & Analysis:** Configurable perturbation types, confidence intervals, and significance testing.
- **Docker Support:** Pre-configured for local or distributed setups.

Quick Start
-----------

The codebase is compatible with **Python 3.12** and **PyTorch 2.5**. GPU support requires appropriate CUDA dependencies.

Install the library via uv (recommended):

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Aleph-Alpha-Research/eval-framework/tree/main
   cd eval-framework

   # Install all dependencies
   uv sync --all-extras

Now, you can run a single benchmark locally:

.. code-block:: bash

   uv run eval_framework \
        --models src/eval_framework/llm/models.py \
        --llm-name Smollm135MInstruct \
        --task-name "MMLU" \
        --task-subjects "abstract_algebra" \
        --output-dir ./eval_results \
        --num-fewshot 5 \
        --num-samples 10

Documentation Overview
----------------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation.md
   cli_usage.md

.. toctree::
   :maxdepth: 1
   :caption: User Guides

   completion_task_guide.md
   add_new_benchmark_guide.md
   benchmarks_and_metrics.md
   controlling_upload_results.md
   docker_guide.md
   evaluate_huggingface_model.md
   loglikelihood_task_guide.md
   model_arguments.md
   overview_dataloading.md
   understanding_results_guide.md
   using_determined.md
   utilities.md
   wandb_integration.md

.. toctree::
   :maxdepth: 1
   :caption: Contributing Guidelines

   ../CONTRIBUTING.md
   testing.md

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index

Citation & License
-----------------

If you use `eval-framework` in research:

.. code-block:: bibtex

    @software{eval_framework,
      title={Aleph Alpha Eval Framework},
      year={2025},
      url={https://github.com/Aleph-Alpha-Research/eval-framework}
    }

Licensed under the [Apache License 2.0](LICENSE).

Funding
-------

This project has received funding from the European Union’s Digital Europe Programme under grant agreement No. 101195233 (OpenEuroLLM). The contents of this publication are the sole responsibility of the OpenEuroLLM consortium.

.. image:: OELLM_1.png
   :width: 100px
   :align: left
   :alt: OELLM Logo 1
   :class: logo

.. image:: OELLM_2.png
   :width: 350px
   :align: left
   :alt: OELLM Logo 2
   :class: logo
