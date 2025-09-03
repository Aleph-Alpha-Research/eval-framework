# PAWSX

````
NAME = PAWSX
DATASET_PATH = google-research-datasets/paws-x
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = validation
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion]
SUBJECTS = ['en', 'de']
LANGUAGE = {'en': <Language.ENG: 'English'>, 'de': <Language.DEU: 'German'>}
````

- Module: [eval_framework.tasks.benchmarks.pawsx](eval_framework.tasks.benchmarks.pawsx)

- File: [src/eval_framework/tasks/benchmarks/pawsx.py](../../src/eval_framework/tasks/benchmarks/pawsx.py)

- Link to dataset: [https://huggingface.co/datasets/google-research-datasets/paws-x](https://huggingface.co/datasets/google-research-datasets/paws-x)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "PAWSX"`.
