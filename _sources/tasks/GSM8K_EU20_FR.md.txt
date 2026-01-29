# GSM8K_EU20_FR

````
NAME = GSM8K_EU20_FR
DATASET_PATH = openGPT-X/gsm8kx
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion]
SUBJECTS = ['FR']
LANGUAGE = <Language.FRA: 'French'>
````

- Module: `eval_framework.tasks.benchmarks.opengptx_eu20`

- File: [src/eval_framework/tasks/benchmarks/opengptx_eu20.py](../../src/eval_framework/tasks/benchmarks/opengptx_eu20.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/opengptx_eu20.py)

- Link to dataset: [https://huggingface.co/datasets/openGPT-X/gsm8kx](https://huggingface.co/datasets/openGPT-X/gsm8kx)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "GSM8K_EU20_FR"`.
