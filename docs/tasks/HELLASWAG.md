# HELLASWAG

````
NAME = HELLASWAG
DATASET_PATH = Rowan/hellaswag
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = train
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.hellaswag](eval_framework.tasks.benchmarks.hellaswag)

- File: [src/eval_framework/tasks/benchmarks/hellaswag.py](../../src/eval_framework/tasks/benchmarks/hellaswag.py)

- Link to dataset: [https://huggingface.co/datasets/Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "HELLASWAG"`.
