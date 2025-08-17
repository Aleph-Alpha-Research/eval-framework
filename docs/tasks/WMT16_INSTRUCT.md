# WMT16_INSTRUCT

````
NAME = WMT16_INSTRUCTDATASET_PATH = wmt16SAMPLE_SPLIT = testFEWSHOT_SPLIT = testRESPONSE_TYPE = COMPLETIONMETRICS = [BLEU, CHRF, TER]SUBJECTS = ['de-en', 'en-de']LANGUAGE = {'de-en': (<Language.DEU: 'German'>, <Language.ENG: 'English'>), 'en-de': (<Language.ENG: 'English'>, <Language.DEU: 'German'>)}````

- Module: [eval_framework.tasks.benchmarks.wmt](eval_framework.tasks.benchmarks.wmt)

- File: [src/eval_framework/tasks/benchmarks/wmt.py](../../src/eval_framework/tasks/benchmarks/wmt.py)

- Link to dataset: [https://huggingface.co/datasets/wmt16](https://huggingface.co/datasets/wmt16)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `poetry run python utils/generate-task-docs.py --add-prompt-examples --only-tasks "WMT16_INSTRUCT"`.
