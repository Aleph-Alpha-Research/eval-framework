# GSM8K_EU20_DE

````
NAME = GSM8K_EU20_DE
DATASET_PATH = openGPT-X/gsm8kx
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion]
SUBJECTS = ['DE']
LANGUAGE = <Language.DEU: 'German'>
````

- Module: [eval_framework.tasks.benchmarks.opengptx_eu20](eval_framework.tasks.benchmarks.opengptx_eu20)

- File: [src/eval_framework/tasks/benchmarks/opengptx_eu20.py](../../src/eval_framework/tasks/benchmarks/opengptx_eu20.py)

- Link to dataset: [https://huggingface.co/datasets/openGPT-X/gsm8kx](https://huggingface.co/datasets/openGPT-X/gsm8kx)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "GSM8K_EU20_DE"`.
