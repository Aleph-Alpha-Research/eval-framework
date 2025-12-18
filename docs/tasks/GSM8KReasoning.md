# GSM8KReasoning

````
NAME = GSM8KReasoning
DATASET_PATH = gsm8k
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion, LanguageRawConsistencyChecker]
SUBJECTS = ['main']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.math_reasoning`

- File: [src/eval_framework/tasks/benchmarks/math_reasoning.py](../../src/eval_framework/tasks/benchmarks/math_reasoning.py) | [View on GitHub](https://github.com/Aleph-Alpha/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/math_reasoning.py)

- Link to dataset: [https://huggingface.co/datasets/gsm8k](https://huggingface.co/datasets/gsm8k)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "GSM8KReasoning"`.
