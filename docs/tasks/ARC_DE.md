# ARC_DE

````
NAME = ARC_DE
DATASET_PATH = LeoLM/ArcChallenge_de
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = validation
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.DEU: 'German'>
````

- Module: [eval_framework.tasks.benchmarks.arc_de](eval_framework.tasks.benchmarks.arc_de)

- File: [src/eval_framework/tasks/benchmarks/arc_de.py](../../src/eval_framework/tasks/benchmarks/arc_de.py)

- Link to dataset: [https://huggingface.co/datasets/LeoLM/ArcChallenge_de](https://huggingface.co/datasets/LeoLM/ArcChallenge_de)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "ARC_DE"`.
