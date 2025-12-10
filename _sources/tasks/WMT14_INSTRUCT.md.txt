# WMT14_INSTRUCT

````
NAME = WMT14_INSTRUCT
DATASET_PATH = wmt14
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = COMPLETION
METRICS = [LINEWISE_BLEU, LINEWISE_CHRF, LINEWISE_TER]
SUBJECTS = ['en-fr', 'fr-en']
LANGUAGE = {'en-fr': (<Language.ENG: 'English'>, <Language.FRA: 'French'>), 'fr-en': (<Language.FRA: 'French'>, <Language.ENG: 'English'>)}
````

- Module: [eval_framework.tasks.benchmarks.wmt](eval_framework.tasks.benchmarks.wmt)

- File: [src/eval_framework/tasks/benchmarks/wmt.py](../../src/eval_framework/tasks/benchmarks/wmt.py)

- Link to dataset: [https://huggingface.co/datasets/wmt14](https://huggingface.co/datasets/wmt14)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "WMT14_INSTRUCT"`.
