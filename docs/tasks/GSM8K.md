# GSM8K

````
NAME = GSM8K
DATASET_PATH = gsm8k
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion]
SUBJECTS = ['main']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.gsm8k](eval_framework.tasks.benchmarks.gsm8k)

- File: [src/eval_framework/tasks/benchmarks/gsm8k.py](../../src/eval_framework/tasks/benchmarks/gsm8k.py)

- Link to dataset: [https://huggingface.co/datasets/gsm8k](https://huggingface.co/datasets/gsm8k)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "GSM8K"`.
