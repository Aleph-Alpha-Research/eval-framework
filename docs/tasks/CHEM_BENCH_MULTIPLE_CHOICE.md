# CHEM_BENCH_MULTIPLE_CHOICE

````
NAME = CHEM_BENCH_MULTIPLE_CHOICE
DATASET_PATH = jablonkagroup/ChemBench
SAMPLE_SPLIT = train
FEWSHOT_SPLIT = train
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['analytical_chemistry', 'chemical_preference', 'general_chemistry', 'inorganic_chemistry', 'materials_science', 'organic_chemistry', 'physical_chemistry', 'technical_chemistry', 'toxicity_and_safety']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.chembench](eval_framework.tasks.benchmarks.chembench)

- File: [src/eval_framework/tasks/benchmarks/chembench.py](../../src/eval_framework/tasks/benchmarks/chembench.py)

- Link to dataset: [https://huggingface.co/datasets/jablonkagroup/ChemBench](https://huggingface.co/datasets/jablonkagroup/ChemBench)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "CHEM_BENCH_MULTIPLE_CHOICE"`.
