# QUALITY

````
NAME = QUALITY
DATASET_PATH = emozilla/quality
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = validation
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['hard', 'easy']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.quality](eval_framework.tasks.benchmarks.quality)

- File: [src/eval_framework/tasks/benchmarks/quality.py](../../src/eval_framework/tasks/benchmarks/quality.py)

- Link to dataset: [https://huggingface.co/datasets/emozilla/quality](https://huggingface.co/datasets/emozilla/quality)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "QUALITY"`.
