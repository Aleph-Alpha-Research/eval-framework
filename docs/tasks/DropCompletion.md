# DropCompletion

````
NAME = DropCompletion
DATASET_PATH = EleutherAI/drop
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = validation
RESPONSE_TYPE = COMPLETION
METRICS = [DropF1ExactMatch]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.drop`

- File: [src/eval_framework/tasks/benchmarks/drop.py](../../src/eval_framework/tasks/benchmarks/drop.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/drop.py)

- Link to dataset: [https://huggingface.co/datasets/EleutherAI/drop](https://huggingface.co/datasets/EleutherAI/drop)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "DropCompletion"`.
