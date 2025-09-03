# INFINITE_BENCH_EN_QA

````
NAME = INFINITE_BENCH_EN_QA
DATASET_PATH = xinrongzhang2022/InfiniteBench
SAMPLE_SPLIT = longbook_qa_eng
FEWSHOT_SPLIT = longbook_qa_eng
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion]
SUBJECTS = ['default']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.infinitebench](eval_framework.tasks.benchmarks.infinitebench)

- File: [src/eval_framework/tasks/benchmarks/infinitebench.py](../../src/eval_framework/tasks/benchmarks/infinitebench.py)

- Link to dataset: [https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "INFINITE_BENCH_EN_QA"`.
