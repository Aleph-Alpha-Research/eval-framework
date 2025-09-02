# FLORES200

````
NAME = FLORES200
DATASET_PATH = facebook/flores
SAMPLE_SPLIT = devtest
FEWSHOT_SPLIT = dev
RESPONSE_TYPE = COMPLETION
METRICS = [BLEU]
SUBJECTS = ['deu_Latn-eng_Latn', 'deu_Latn-fin_Latn', 'deu_Latn-fra_Latn', 'deu_Latn-nld_Latn', 'eng_Latn-deu_Latn', 'eng_Latn-fin_Latn', 'eng_Latn-fra_Latn', 'eng_Latn-nld_Latn', 'fin_Latn-deu_Latn', 'fin_Latn-eng_Latn', 'fin_Latn-fra_Latn', 'fin_Latn-nld_Latn', 'fra_Latn-deu_Latn', 'fra_Latn-eng_Latn', 'fra_Latn-fin_Latn', 'fra_Latn-nld_Latn', 'nld_Latn-deu_Latn', 'nld_Latn-eng_Latn', 'nld_Latn-fin_Latn', 'nld_Latn-fra_Latn']
LANGUAGE = {'deu_Latn': <Language.DEU: 'German'>, 'eng_Latn': <Language.ENG: 'English'>, 'fin_Latn': <Language.FIN: 'Finnish'>, 'fra_Latn': <Language.FRA: 'French'>, 'nld_Latn': <Language.NLD: 'Dutch'>}
````

- Module: [eval_framework.tasks.benchmarks.flores200](eval_framework.tasks.benchmarks.flores200)

- File: [src/eval_framework/tasks/benchmarks/flores200.py](../../src/eval_framework/tasks/benchmarks/flores200.py)

- Link to dataset: [https://huggingface.co/datasets/facebook/flores](https://huggingface.co/datasets/facebook/flores)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/utils/generate_task_docs.py --add-prompt-examples --only-tasks "FLORES200"`.
