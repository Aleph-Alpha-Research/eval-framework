# ZERO_SCROLLS_QUALITY

````
NAME = ZERO_SCROLLS_QUALITY
DATASET_PATH = tau/zero_scrolls
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = validation
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood]
SUBJECTS = ['quality']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.zero_scrolls](eval_framework.tasks.benchmarks.zero_scrolls)

- File: [src/eval_framework/tasks/benchmarks/zero_scrolls.py](../../src/eval_framework/tasks/benchmarks/zero_scrolls.py)

- Link to dataset: [https://huggingface.co/datasets/tau/zero_scrolls](https://huggingface.co/datasets/tau/zero_scrolls)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "ZERO_SCROLLS_QUALITY"`.
