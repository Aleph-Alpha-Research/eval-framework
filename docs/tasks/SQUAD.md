# SQUAD

````
NAME = SQUAD
DATASET_PATH = rajpurkar/squad
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion, F1]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.squad](eval_framework.tasks.benchmarks.squad)

- File: [src/eval_framework/tasks/benchmarks/squad.py](../../src/eval_framework/tasks/benchmarks/squad.py)

- Link to dataset: [https://huggingface.co/datasets/rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "SQUAD"`.
