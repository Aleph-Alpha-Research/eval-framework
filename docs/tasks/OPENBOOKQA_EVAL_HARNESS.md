# OPENBOOKQA_EVAL_HARNESS

````
NAME = OPENBOOKQA_EVAL_HARNESS
DATASET_PATH = allenai/openbookqa
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = test
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['additional']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.openbookqa](eval_framework.tasks.benchmarks.openbookqa)

- File: [src/eval_framework/tasks/benchmarks/openbookqa.py](../../src/eval_framework/tasks/benchmarks/openbookqa.py)

- Link to dataset: [https://huggingface.co/datasets/allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "OPENBOOKQA_EVAL_HARNESS"`.
