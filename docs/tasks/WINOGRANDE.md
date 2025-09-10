# WINOGRANDE

````
NAME = WINOGRANDE
DATASET_PATH = winogrande
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = train
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['winogrande_xl']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.winogrande](eval_framework.tasks.benchmarks.winogrande)

- File: [src/eval_framework/tasks/benchmarks/winogrande.py](../../src/eval_framework/tasks/benchmarks/winogrande.py)

- Link to dataset: [https://huggingface.co/datasets/winogrande](https://huggingface.co/datasets/winogrande)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "WINOGRANDE"`.
