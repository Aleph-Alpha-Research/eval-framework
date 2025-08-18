# Utils in `eval-framework`

## Plot results

A basic utility to plot evaluation results from a set of JSON files is provided as example in `utils/plot-results.py`.
This script reads JSON files containing evaluation results, filters them based on specified criteria, and generates a
plot using Matplotlib. This script can be adjusted to your plotting need and will produce a basic plot if run with:
```bash
# loop over all tasks and models under a given parent folder
poetry run python utils/plot-results.py --folder PARENT_RESULTS_FOLDER
```

More CLI arguments are available and can be listed with `poetry run python utils/plot-results.py --help`.

## Inspect JSON results

The detailed results and completions for each sample are saved as a JSONL. To help inspecting this file a basic utility
script is provided that print the content of the file, split the line in a readable way and colorize the output.

For example:
```bash
poetry run python utils/inspect-jsonl.py output.jsonl --highlight prompt,completion --strip messages,eval_kwargs,raw_completion
```

Use `poetry run python utils/inspect-jsonl.py --help` to get all CLI arguments.

## Document benchmark tasks

The `utils/generate-task-docs.py` can be use to update or further detail the automated task description. This script is discussed in
[docs/installation.md](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/docs/installation.md#generate-task-documentation).
