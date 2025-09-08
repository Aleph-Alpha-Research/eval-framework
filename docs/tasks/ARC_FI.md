# ARC_FI

````
NAME = ARC_FI
DATASET_PATH = LumiOpen/arc_challenge_mt
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = validation
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['fi']
LANGUAGE = <Language.FIN: 'Finnish'>
````

- Module: [eval_framework.tasks.benchmarks.arc_fi](eval_framework.tasks.benchmarks.arc_fi)

- File: [src/eval_framework/tasks/benchmarks/arc_fi.py](../../src/eval_framework/tasks/benchmarks/arc_fi.py)

- Link to dataset: [https://huggingface.co/datasets/LumiOpen/arc_challenge_mt](https://huggingface.co/datasets/LumiOpen/arc_challenge_mt)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/utils/generate_task_docs.py --add-prompt-examples --only-tasks "ARC_FI"`.
