# InfiniteBench_EnQA

````
NAME = InfiniteBench_EnQA
DATASET_PATH = xinrongzhang2022/InfiniteBench
SAMPLE_SPLIT = longbook_qa_eng
FEWSHOT_SPLIT = longbook_qa_eng
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion]
SUBJECTS = ['default']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.infinitebench`

- File: [src/eval_framework/tasks/benchmarks/infinitebench.py](../../src/eval_framework/tasks/benchmarks/infinitebench.py) | [View on GitHub](https://github.com/Aleph-Alpha/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/infinitebench.py)

- Link to dataset: [https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "InfiniteBench_EnQA"`.
