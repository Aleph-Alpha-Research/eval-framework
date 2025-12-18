# TRUTHFULQA_EU20_FR

````
NAME = TRUTHFULQA_EU20_FR
DATASET_PATH = openGPT-X/truthfulqax
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT =
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, ProbabilityMass, ProbabilityMassNorm]
SUBJECTS = ['mc1', 'mc2']
LANGUAGE = <Language.FRA: 'French'>
````

- Module: `eval_framework.tasks.benchmarks.opengptx_eu20`

- File: [src/eval_framework/tasks/benchmarks/opengptx_eu20.py](../../src/eval_framework/tasks/benchmarks/opengptx_eu20.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/opengptx_eu20.py)

- Link to dataset: [https://huggingface.co/datasets/openGPT-X/truthfulqax](https://huggingface.co/datasets/openGPT-X/truthfulqax)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "TRUTHFULQA_EU20_FR"`.
