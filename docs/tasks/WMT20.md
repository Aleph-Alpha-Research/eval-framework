# WMT20

````
NAME = WMT20DATASET_PATH = wmt20SAMPLE_SPLIT = testFEWSHOT_SPLIT = testRESPONSE_TYPE = COMPLETIONMETRICS = [BLEU, CHRF, TER]SUBJECTS = ['de-en', 'de-fr', 'en-de', 'fr-de']LANGUAGE = {'de-en': (<Language.DEU: 'German'>, <Language.ENG: 'English'>), 'de-fr': (<Language.DEU: 'German'>, <Language.FRA: 'French'>), 'en-de': (<Language.ENG: 'English'>, <Language.DEU: 'German'>), 'fr-de': (<Language.FRA: 'French'>, <Language.DEU: 'German'>)}````

- Module: [eval_framework.tasks.benchmarks.wmt](eval_framework.tasks.benchmarks.wmt)

- File: [src/eval_framework/tasks/benchmarks/wmt.py](../../src/eval_framework/tasks/benchmarks/wmt.py)

- Link to dataset: [https://huggingface.co/datasets/wmt20](https://huggingface.co/datasets/wmt20)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python utils/generate-task-docs.py --add-prompt-examples --only-tasks "WMT20"`.
