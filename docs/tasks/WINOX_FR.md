# WINOX_FR

````
NAME = WINOX_FR
DATASET_PATH = demelin/wino_x
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['lm_en_fr']
LANGUAGE = <Language.FRA: 'French'>
````

- Module: [eval_framework.tasks.benchmarks.winox](eval_framework.tasks.benchmarks.winox)

- File: [src/eval_framework/tasks/benchmarks/winox.py](../../src/eval_framework/tasks/benchmarks/winox.py)

- Link to dataset: [https://huggingface.co/datasets/demelin/wino_x](https://huggingface.co/datasets/demelin/wino_x)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/utils/generate_task_docs.py --add-prompt-examples --only-tasks "WINOX_FR"`.
