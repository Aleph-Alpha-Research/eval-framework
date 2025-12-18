# MBPP_PROMPT_WITHOUT_TESTS_SANITIZED

````
NAME = MBPP_PROMPT_WITHOUT_TESTS_SANITIZED
DATASET_PATH = google-research-datasets/mbpp
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [CodeCompletionAssertion]
SUBJECTS = ['sanitized']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.mbpp`

- File: [src/eval_framework/tasks/benchmarks/mbpp.py](../../src/eval_framework/tasks/benchmarks/mbpp.py) | [View on GitHub](https://github.com/Aleph-Alpha/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/mbpp.py)

- Link to dataset: [https://huggingface.co/datasets/google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "MBPP_PROMPT_WITHOUT_TESTS_SANITIZED"`.
