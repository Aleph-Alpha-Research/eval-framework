# ARC

````
NAME = ARC
DATASET_PATH = ai2_arc
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['ARC-Easy', 'ARC-Challenge']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.arc](eval_framework.tasks.benchmarks.arc)

- File: [src/eval_framework/tasks/benchmarks/arc.py](../../src/eval_framework/tasks/benchmarks/arc.py)

- Link to dataset: [https://huggingface.co/datasets/ai2_arc](https://huggingface.co/datasets/ai2_arc)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "ARC"`.
