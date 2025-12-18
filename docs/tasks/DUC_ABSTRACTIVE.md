# DUC_ABSTRACTIVE

````
NAME = DUC_ABSTRACTIVE
DATASET_PATH = midas/duc2001
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion]
SUBJECTS = ['raw']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.duc`

- File: [src/eval_framework/tasks/benchmarks/duc.py](../../src/eval_framework/tasks/benchmarks/duc.py) | [View on GitHub](https://github.com/Aleph-Alpha/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/duc.py)

- Link to dataset: [https://huggingface.co/datasets/midas/duc2001](https://huggingface.co/datasets/midas/duc2001)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "DUC_ABSTRACTIVE"`.
