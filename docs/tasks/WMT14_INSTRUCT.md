# WMT14_INSTRUCT

````
NAME = WMT14_INSTRUCTDATASET_PATH = wmt14SAMPLE_SPLIT = testFEWSHOT_SPLIT = testRESPONSE_TYPE = COMPLETIONMETRICS = [BLEU, CHRF, TER]SUBJECTS = ['en-fr', 'fr-en']LANGUAGE = {'en-fr': (<Language.ENG: 'English'>, <Language.FRA: 'French'>), 'fr-en': (<Language.FRA: 'French'>, <Language.ENG: 'English'>)}````

- Module: [eval_framework.tasks.benchmarks.wmt](eval_framework.tasks.benchmarks.wmt)

- File: [src/eval_framework/tasks/benchmarks/wmt.py](../../src/eval_framework/tasks/benchmarks/wmt.py)

- Link to dataset: [https://huggingface.co/datasets/wmt14](https://huggingface.co/datasets/wmt14)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python utils/generate-task-docs.py --add-prompt-examples --only-tasks "WMT14_INSTRUCT"`.
