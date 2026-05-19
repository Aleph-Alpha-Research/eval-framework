# GPQA

````
NAME = GPQA
DATASET_PATH = Idavidrein/gpqa
SAMPLE_SPLIT = train
FEWSHOT_SPLIT = train
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['gpqa_extended']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.gpqa`

- File: [src/eval_framework/tasks/benchmarks/gpqa.py](../../src/eval_framework/tasks/benchmarks/gpqa.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/gpqa.py)

- Link to dataset: [https://huggingface.co/datasets/Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "GPQA"`.
