# GOLDENSWAG

````
NAME = GOLDENSWAG
DATASET_PATH = PleIAs/GoldenSwag
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = validation
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.goldenswag`

- File: [src/eval_framework/tasks/benchmarks/goldenswag.py](../../src/eval_framework/tasks/benchmarks/goldenswag.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/goldenswag.py)

- Link to dataset: [https://huggingface.co/datasets/PleIAs/GoldenSwag](https://huggingface.co/datasets/PleIAs/GoldenSwag)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "GOLDENSWAG"`.
