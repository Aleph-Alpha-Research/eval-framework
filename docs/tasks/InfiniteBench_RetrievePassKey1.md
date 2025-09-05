# InfiniteBench_RetrievePassKey1

````
NAME = InfiniteBench_RetrievePassKey1
DATASET_PATH = xinrongzhang2022/InfiniteBench
SAMPLE_SPLIT = passkey
FEWSHOT_SPLIT = passkey
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion]
SUBJECTS = ['default']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.infinitebench](eval_framework.tasks.benchmarks.infinitebench)

- File: [src/eval_framework/tasks/benchmarks/infinitebench.py](../../src/eval_framework/tasks/benchmarks/infinitebench.py)

- Link to dataset: [https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/utils/generate_task_docs.py --add-prompt-examples --only-tasks "InfiniteBench_RetrievePassKey1"`.
