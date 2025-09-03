# MMLU_PRO_COT

````
NAME = MMLU_PRO_COT
DATASET_PATH = TIGER-Lab/MMLU-Pro
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion]
SUBJECTS = ['engineering', 'physics', 'psychology', 'chemistry', 'biology', 'law', 'philosophy', 'computer science', 'other', 'economics', 'business', 'history', 'math', 'health']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.mmlu_pro](eval_framework.tasks.benchmarks.mmlu_pro)

- File: [src/eval_framework/tasks/benchmarks/mmlu_pro.py](../../src/eval_framework/tasks/benchmarks/mmlu_pro.py)

- Link to dataset: [https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/utils/generate_task_docs.py --add-prompt-examples --only-tasks "MMLU_PRO_COT"`.
