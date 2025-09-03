# IFEVAL

````
NAME = IFEVAL
DATASET_PATH = google/IFEval
SAMPLE_SPLIT = train
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [IFEvalMetric]
SUBJECTS = ['no_subject']
LANGUAGE = {'no_subject': <Language.ENG: 'English'>}
````

- Module: [eval_framework.tasks.benchmarks.ifeval](eval_framework.tasks.benchmarks.ifeval)

- File: [src/eval_framework/tasks/benchmarks/ifeval.py](../../src/eval_framework/tasks/benchmarks/ifeval.py)

- Link to dataset: [https://huggingface.co/datasets/google/IFEval](https://huggingface.co/datasets/google/IFEval)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "IFEVAL"`.
