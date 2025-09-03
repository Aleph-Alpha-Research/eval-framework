# FLORES_PLUS

````
NAME = FLORES_PLUS
DATASET_PATH = openlanguagedata/flores_plus
SAMPLE_SPLIT = dev
FEWSHOT_SPLIT = devtest
RESPONSE_TYPE = COMPLETION
METRICS = [BLEU, CHRF, COMET]
SUBJECTS = ['deu_Latn-eng_Latn', 'deu_Latn-fra_Latn', 'deu_Latn-ita_Latn', 'deu_Latn-nld_Latn', 'deu_Latn-pol_Latn', 'deu_Latn-rus_Cyrl', 'deu_Latn-spa_Latn', 'deu_Latn-ukr_Cyrl', 'eng_Latn-deu_Latn', 'eng_Latn-fra_Latn', 'eng_Latn-ita_Latn', 'eng_Latn-nld_Latn', 'eng_Latn-pol_Latn', 'eng_Latn-rus_Cyrl', 'eng_Latn-spa_Latn', 'eng_Latn-ukr_Cyrl', 'fra_Latn-deu_Latn', 'fra_Latn-eng_Latn', 'fra_Latn-ita_Latn', 'fra_Latn-nld_Latn', 'fra_Latn-pol_Latn', 'fra_Latn-rus_Cyrl', 'fra_Latn-spa_Latn', 'fra_Latn-ukr_Cyrl', 'ita_Latn-deu_Latn', 'ita_Latn-eng_Latn', 'ita_Latn-fra_Latn', 'ita_Latn-nld_Latn', 'ita_Latn-pol_Latn', 'ita_Latn-rus_Cyrl', 'ita_Latn-spa_Latn', 'ita_Latn-ukr_Cyrl', 'nld_Latn-deu_Latn', 'nld_Latn-eng_Latn', 'nld_Latn-fra_Latn', 'nld_Latn-ita_Latn', 'nld_Latn-pol_Latn', 'nld_Latn-rus_Cyrl', 'nld_Latn-spa_Latn', 'nld_Latn-ukr_Cyrl', 'pol_Latn-deu_Latn', 'pol_Latn-eng_Latn', 'pol_Latn-fra_Latn', 'pol_Latn-ita_Latn', 'pol_Latn-nld_Latn', 'pol_Latn-rus_Cyrl', 'pol_Latn-spa_Latn', 'pol_Latn-ukr_Cyrl', 'rus_Cyrl-deu_Latn', 'rus_Cyrl-eng_Latn', 'rus_Cyrl-fra_Latn', 'rus_Cyrl-ita_Latn', 'rus_Cyrl-nld_Latn', 'rus_Cyrl-pol_Latn', 'rus_Cyrl-spa_Latn', 'rus_Cyrl-ukr_Cyrl', 'spa_Latn-deu_Latn', 'spa_Latn-eng_Latn', 'spa_Latn-fra_Latn', 'spa_Latn-ita_Latn', 'spa_Latn-nld_Latn', 'spa_Latn-pol_Latn', 'spa_Latn-rus_Cyrl', 'spa_Latn-ukr_Cyrl', 'ukr_Cyrl-deu_Latn', 'ukr_Cyrl-eng_Latn', 'ukr_Cyrl-fra_Latn', 'ukr_Cyrl-ita_Latn', 'ukr_Cyrl-nld_Latn', 'ukr_Cyrl-pol_Latn', 'ukr_Cyrl-rus_Cyrl', 'ukr_Cyrl-spa_Latn']
LANGUAGE = {'deu_Latn': <Language.DEU: 'German'>, 'eng_Latn': <Language.ENG: 'English'>, 'fra_Latn': <Language.FRA: 'French'>, 'ita_Latn': <Language.ITA: 'Italian'>, 'nld_Latn': <Language.NLD: 'Dutch'>, 'pol_Latn': <Language.POL: 'Polish'>, 'rus_Cyrl': <Language.RUS: 'Russian'>, 'spa_Latn': <Language.SPA: 'Spanish'>, 'ukr_Cyrl': <Language.UKR: 'Ukrainian'>}
````

- Module: [eval_framework.tasks.benchmarks.flores_plus](eval_framework.tasks.benchmarks.flores_plus)

- File: [src/eval_framework/tasks/benchmarks/flores_plus.py](../../src/eval_framework/tasks/benchmarks/flores_plus.py)

- Link to dataset: [https://huggingface.co/datasets/openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "FLORES_PLUS"`.
