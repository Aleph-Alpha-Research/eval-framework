Eval-Framework Documentation
============================

A suite of curated benchmarks for evaluating **L**arge **L**anguage **M**odels (LLMs)

Benchmarks in this suite are:

- Validated: Benchmarks implemented in this suite are required to have reproduced published results
- Legal: The data is legal to exist and to use. Some datasets are subject to third party licenses
- Comprehensive: For a list of supported benchmarks execute the CLI tool or check the benchmarks
  source directory. https://github.com/Aleph-Alpha-Research/eval-framework/tree/main/src/eval_framework/benchmarks


Quick Start
-----------

Running a benchmark locally.

Clone the repository

.. code-block:: bash

    git clone https://github.com/Aleph-Alpha-Research/eval-framework/tree/main
    cd eval-framework

Setup the Python environment. We recommend using `uv https://docs.astral.sh/uv/getting-started/installation/`:

.. code-block:: bash

   uv sync

By default Open AIs API is used. You can explicitly specify an OpenAI compatible endpoint by
passing ``base_url=http://host:port/v1`` together with the `--llm-args` options. Some metrics make
use of the additional information provided by vLLM and won't work with OpenAI. In this exapmle we
stick with the default. Therfore the ``OPENAI_API_KEY`` environment variable needs to contain an API
token.

.. code-block:: bash

   uv run eval_framework \
        --llm-args model_name="gpt-4o-mini" \
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

   naming_convention.md
   benchmarks_and_metrics.md
   controlling_upload_results.md
   docker_guide.md
   model_arguments.md
   overview_dataloading.md
   understanding_results_guide.md
   utilities.md
   wandb_integration.md

.. toctree::
   :maxdepth: 1
   :caption: Testing Guidelines

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

This project has received funding from the European Union’s Digital Europe Programme under grant
agreement No. 101195233 (OpenEuroLLM). The contents of this publication are the sole responsibility
of the OpenEuroLLM consortium.

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
