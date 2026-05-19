# AidanBench

````
NAME = AidanBench
DATASET_PATH = Aleph-Alpha-Research/aidanbench
SAMPLE_SPLIT = train
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [AidanBenchMetric]
SUBJECTS = ['no_subject']
LANGUAGE = {'no_subject': <Language.ENG: 'English'>}
````

- Module: `eval_framework.tasks.benchmarks.aidanbench`

- File: [src/eval_framework/tasks/benchmarks/aidanbench.py](../../src/eval_framework/tasks/benchmarks/aidanbench.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/aidanbench.py)

- Link to dataset: [https://huggingface.co/datasets/Aleph-Alpha-Research/aidanbench](https://huggingface.co/datasets/Aleph-Alpha-Research/aidanbench)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "AidanBench"`.
