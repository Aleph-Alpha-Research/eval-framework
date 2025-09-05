# BigCodeBench

````
NAME = BigCodeBench
DATASET_PATH = bigcode/bigcodebench
SAMPLE_SPLIT = v0.1.4
FEWSHOT_SPLIT = v0.1.4
RESPONSE_TYPE = COMPLETION
METRICS = [CodeExecutionPassAtOne]
SUBJECTS = ['original', 'calibrated']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.bigcodebench](eval_framework.tasks.benchmarks.bigcodebench)

- File: [src/eval_framework/tasks/benchmarks/bigcodebench.py](../../src/eval_framework/tasks/benchmarks/bigcodebench.py)

- Link to dataset: [https://huggingface.co/datasets/bigcode/bigcodebench](https://huggingface.co/datasets/bigcode/bigcodebench)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/utils/generate_task_docs.py --add-prompt-examples --only-tasks "BigCodeBench"`.
