# HumanEvalInstruct

````
NAME = HumanEvalInstruct
DATASET_PATH = openai/openai_humaneval
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = COMPLETION
METRICS = [CodeCompletionAssertion]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.humaneval](eval_framework.tasks.benchmarks.humaneval)

- File: [src/eval_framework/tasks/benchmarks/humaneval.py](../../src/eval_framework/tasks/benchmarks/humaneval.py)

- Link to dataset: [https://huggingface.co/datasets/openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "HumanEvalInstruct"`.
