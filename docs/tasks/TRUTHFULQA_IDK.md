# TRUTHFULQA_IDK

````
NAME = TRUTHFULQA_IDK
DATASET_PATH = truthful_qa
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT =
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, ConfidenceWeightedAccuracy, DistributionalCorrectnessScore, TernaryScore]
SUBJECTS = ['mc1', 'mc2']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.truthfulqa`

- File: [src/eval_framework/tasks/benchmarks/truthfulqa.py](../../src/eval_framework/tasks/benchmarks/truthfulqa.py) | [View on GitHub](https://github.com/Aleph-Alpha/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/truthfulqa.py)

- Link to dataset: [https://huggingface.co/datasets/truthful_qa](https://huggingface.co/datasets/truthful_qa)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "TRUTHFULQA_IDK"`.
