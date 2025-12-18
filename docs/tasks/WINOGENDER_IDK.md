# WINOGENDER_IDK

````
NAME = WINOGENDER_IDK
DATASET_PATH = oskarvanderwal/winogender
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, ConfidenceWeightedAccuracy, DistributionalCorrectnessScore, TernaryScore]
SUBJECTS = ['all']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.winogender`

- File: [src/eval_framework/tasks/benchmarks/winogender.py](../../src/eval_framework/tasks/benchmarks/winogender.py) | [View on GitHub](https://github.com/Aleph-Alpha/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/winogender.py)

- Link to dataset: [https://huggingface.co/datasets/oskarvanderwal/winogender](https://huggingface.co/datasets/oskarvanderwal/winogender)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "WINOGENDER_IDK"`.
