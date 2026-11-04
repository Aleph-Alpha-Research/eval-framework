# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
It states only the norms the code cannot express itself. For everything else, read the code:
`src/eval_framework/benchmarks/` shows how benchmarks are built, `pyproject.toml` holds the tooling
(pytest markers, ruff, mypy), and `docs/` the user guides.

- A changed hash in `tests/tests_eval_framework/benchmarks/task-prompts-hashes.json` means the prompts
  changed and results are no longer comparable to earlier runs. Acceptable only when provably pure
  sample reordering; otherwise treat it as breaking the benchmark.
- New benchmark ids must follow `docs/naming_convention.md`.
- New benchmarks pin their dataset: add the revision to
  `src/eval_framework/tasks/hf-dataset-revisions.json` and load through `pinned_by_framework`.
- Commit messages follow [conventional commits](https://www.conventionalcommits.org/) — release-please
  derives version bumps and release notes from them.
- The CLI always uses `OpenAIModel`; `model_name` is required in `--llm-args` and, for an LLM judge,
  in `--judge-model-args`.
- Some tests read gated Hugging Face repos (e.g. the Llama tokenizers) and need an authorized
  `HF_TOKEN` in the environment.
