# WINOX_DE

````
NAME = WINOX_DE
DATASET_PATH = demelin/wino_x
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['lm_en_de']
LANGUAGE = <Language.DEU: 'German'>
````

- Module: `eval_framework.tasks.benchmarks.winox`

- File: [src/eval_framework/tasks/benchmarks/winox.py](../../src/eval_framework/tasks/benchmarks/winox.py) | [View on GitHub](https://github.com/Aleph-Alpha/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/winox.py)

- Link to dataset: [https://huggingface.co/datasets/demelin/wino_x](https://huggingface.co/datasets/demelin/wino_x)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "WINOX_DE"`.
