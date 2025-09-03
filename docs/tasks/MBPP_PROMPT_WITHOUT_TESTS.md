# MBPP_PROMPT_WITHOUT_TESTS

````
NAME = MBPP_PROMPT_WITHOUT_TESTS
DATASET_PATH = google-research-datasets/mbpp
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [CodeCompletionAssertion]
SUBJECTS = ['full']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.mbpp](eval_framework.tasks.benchmarks.mbpp)

- File: [src/eval_framework/tasks/benchmarks/mbpp.py](../../src/eval_framework/tasks/benchmarks/mbpp.py)

- Link to dataset: [https://huggingface.co/datasets/google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "MBPP_PROMPT_WITHOUT_TESTS"`.
