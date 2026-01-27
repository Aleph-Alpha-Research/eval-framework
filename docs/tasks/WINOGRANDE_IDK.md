# WINOGRANDE_IDK

````
NAME = WINOGRANDE_IDK
DATASET_PATH = allenai/winogrande
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = train
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, ConfidenceWeightedAccuracy, DistributionalCorrectnessScore, TernaryScore]
SUBJECTS = ['winogrande_xl']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.winogrande`

- File: [src/eval_framework/tasks/benchmarks/winogrande.py](../../src/eval_framework/tasks/benchmarks/winogrande.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/winogrande.py)

- Link to dataset: [https://huggingface.co/datasets/allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "WINOGRANDE_IDK"`.
