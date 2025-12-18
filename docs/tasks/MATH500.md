# MATH500

````
NAME = MATH500
DATASET_PATH = HuggingFaceH4/MATH-500
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = COMPLETION
METRICS = [MathReasoningCompletion, LanguageRawConsistencyChecker]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.math_reasoning`

- File: [src/eval_framework/tasks/benchmarks/math_reasoning.py](../../src/eval_framework/tasks/benchmarks/math_reasoning.py) | [View on GitHub](https://github.com/Aleph-Alpha/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/math_reasoning.py)

- Link to dataset: [https://huggingface.co/datasets/HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "MATH500"`.
