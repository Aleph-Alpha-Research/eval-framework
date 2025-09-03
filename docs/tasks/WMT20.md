# WMT20

````
NAME = WMT20
DATASET_PATH = wmt20
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = COMPLETION
METRICS = [LINEWISE_BLEU, LINEWISE_CHRF, LINEWISE_TER]
SUBJECTS = ['de-en', 'de-fr', 'en-de', 'fr-de']
LANGUAGE = {'de-en': (<Language.DEU: 'German'>, <Language.ENG: 'English'>), 'de-fr': (<Language.DEU: 'German'>, <Language.FRA: 'French'>), 'en-de': (<Language.ENG: 'English'>, <Language.DEU: 'German'>), 'fr-de': (<Language.FRA: 'French'>, <Language.DEU: 'German'>)}
````

- Module: [eval_framework.tasks.benchmarks.wmt](eval_framework.tasks.benchmarks.wmt)

- File: [src/eval_framework/tasks/benchmarks/wmt.py](../../src/eval_framework/tasks/benchmarks/wmt.py)

- Link to dataset: [https://huggingface.co/datasets/wmt20](https://huggingface.co/datasets/wmt20)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "WMT20"`.
