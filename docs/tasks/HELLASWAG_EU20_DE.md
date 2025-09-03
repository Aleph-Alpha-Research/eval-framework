# HELLASWAG_EU20_DE

````
NAME = HELLASWAG_EU20_DE
DATASET_PATH = openGPT-X/hellaswagx
SAMPLE_SPLIT = train
FEWSHOT_SPLIT = validation
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['DE']
LANGUAGE = <Language.DEU: 'German'>
````

- Module: [eval_framework.tasks.benchmarks.opengptx_eu20](eval_framework.tasks.benchmarks.opengptx_eu20)

- File: [src/eval_framework/tasks/benchmarks/opengptx_eu20.py](../../src/eval_framework/tasks/benchmarks/opengptx_eu20.py)

- Link to dataset: [https://huggingface.co/datasets/openGPT-X/hellaswagx](https://huggingface.co/datasets/openGPT-X/hellaswagx)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "HELLASWAG_EU20_DE"`.
