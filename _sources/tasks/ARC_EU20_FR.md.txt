# ARC_EU20_FR

````
NAME = ARC_EU20_FR
DATASET_PATH = openGPT-X/arcx
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['challenge_FR', 'easy_FR']
LANGUAGE = <Language.FRA: 'French'>
````

- Module: [eval_framework.tasks.benchmarks.opengptx_eu20](eval_framework.tasks.benchmarks.opengptx_eu20)

- File: [src/eval_framework/tasks/benchmarks/opengptx_eu20.py](../../src/eval_framework/tasks/benchmarks/opengptx_eu20.py)

- Link to dataset: [https://huggingface.co/datasets/openGPT-X/arcx](https://huggingface.co/datasets/openGPT-X/arcx)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "ARC_EU20_FR"`.
