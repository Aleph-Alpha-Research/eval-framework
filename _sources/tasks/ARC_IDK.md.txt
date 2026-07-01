# ARC_IDK

````
NAME = ARC_IDK
DATASET_PATH = allenai/ai2_arc
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, ConfidenceWeightedAccuracy, DistributionalCorrectnessScore, TernaryScore]
SUBJECTS = ['ARC-Easy', 'ARC-Challenge']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.arc`

- File: [src/eval_framework/tasks/benchmarks/arc.py](../../src/eval_framework/tasks/benchmarks/arc.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/arc.py)

- Link to dataset: [https://huggingface.co/datasets/allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "ARC_IDK"`.
