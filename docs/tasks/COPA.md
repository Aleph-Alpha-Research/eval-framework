# COPA

````
NAME = COPA
DATASET_PATH = aps/super_glue
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = test
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['copa']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.copa](eval_framework.tasks.benchmarks.copa)

- File: [src/eval_framework/tasks/benchmarks/copa.py](../../src/eval_framework/tasks/benchmarks/copa.py)

- Link to dataset: [https://huggingface.co/datasets/aps/super_glue](https://huggingface.co/datasets/aps/super_glue)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "COPA"`.
