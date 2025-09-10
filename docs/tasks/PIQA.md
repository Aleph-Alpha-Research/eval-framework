# PIQA

````
NAME = PIQA
DATASET_PATH = ybisk/piqa
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = test
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.piqa](eval_framework.tasks.benchmarks.piqa)

- File: [src/eval_framework/tasks/benchmarks/piqa.py](../../src/eval_framework/tasks/benchmarks/piqa.py)

- Link to dataset: [https://huggingface.co/datasets/ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "PIQA"`.
