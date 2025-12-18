# WMT16_INSTRUCT

````
NAME = WMT16_INSTRUCT
DATASET_PATH = wmt16
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = COMPLETION
METRICS = [LINEWISE_BLEU, LINEWISE_CHRF, LINEWISE_TER]
SUBJECTS = ['de-en', 'en-de']
LANGUAGE = {'de-en': (<Language.DEU: 'German'>, <Language.ENG: 'English'>), 'en-de': (<Language.ENG: 'English'>, <Language.DEU: 'German'>)}
````

- Module: `eval_framework.tasks.benchmarks.wmt`

- File: [src/eval_framework/tasks/benchmarks/wmt.py](../../src/eval_framework/tasks/benchmarks/wmt.py) | [View on GitHub](https://github.com/Aleph-Alpha/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/wmt.py)

- Link to dataset: [https://huggingface.co/datasets/wmt16](https://huggingface.co/datasets/wmt16)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "WMT16_INSTRUCT"`.
