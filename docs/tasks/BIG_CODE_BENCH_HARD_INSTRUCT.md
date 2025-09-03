# BIG_CODE_BENCH_HARD_INSTRUCT

````
NAME = BIG_CODE_BENCH_HARD_INSTRUCT
DATASET_PATH = bigcode/bigcodebench-hard
SAMPLE_SPLIT = v0.1.4
FEWSHOT_SPLIT = v0.1.4
RESPONSE_TYPE = COMPLETION
METRICS = [CodeExecutionPassAtOne]
SUBJECTS = ['original', 'calibrated']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.bigcodebench](eval_framework.tasks.benchmarks.bigcodebench)

- File: [src/eval_framework/tasks/benchmarks/bigcodebench.py](../../src/eval_framework/tasks/benchmarks/bigcodebench.py)

- Link to dataset: [https://huggingface.co/datasets/bigcode/bigcodebench-hard](https://huggingface.co/datasets/bigcode/bigcodebench-hard)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "BIG_CODE_BENCH_HARD_INSTRUCT"`.
