# MMMLU_German

````
NAME = MMMLU_German
DATASET_PATH = openai/MMMLU
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = [('DE_DE', 'abstract_algebra'), ('DE_DE', 'astronomy'), ('DE_DE', 'college_biology'), ('DE_DE', 'college_chemistry'), ('DE_DE', 'college_computer_science'), ('DE_DE', 'college_mathematics'), ('DE_DE', 'college_physics'), ('DE_DE', 'computer_security'), ('DE_DE', 'conceptual_physics'), ('DE_DE', 'electrical_engineering'), ('DE_DE', 'elementary_mathematics'), ('DE_DE', 'high_school_biology'), ('DE_DE', 'high_school_chemistry'), ('DE_DE', 'high_school_computer_science'), ('DE_DE', 'high_school_mathematics'), ('DE_DE', 'high_school_physics'), ('DE_DE', 'high_school_statistics'), ('DE_DE', 'machine_learning'), ('DE_DE', 'formal_logic'), ('DE_DE', 'high_school_european_history'), ('DE_DE', 'high_school_us_history'), ('DE_DE', 'high_school_world_history'), ('DE_DE', 'international_law'), ('DE_DE', 'jurisprudence'), ('DE_DE', 'logical_fallacies'), ('DE_DE', 'moral_disputes'), ('DE_DE', 'moral_scenarios'), ('DE_DE', 'philosophy'), ('DE_DE', 'prehistory'), ('DE_DE', 'professional_law'), ('DE_DE', 'world_religions'), ('DE_DE', 'econometrics'), ('DE_DE', 'high_school_geography'), ('DE_DE', 'high_school_government_and_politics'), ('DE_DE', 'high_school_macroeconomics'), ('DE_DE', 'high_school_microeconomics'), ('DE_DE', 'high_school_psychology'), ('DE_DE', 'human_sexuality'), ('DE_DE', 'professional_psychology'), ('DE_DE', 'public_relations'), ('DE_DE', 'security_studies'), ('DE_DE', 'sociology'), ('DE_DE', 'us_foreign_policy'), ('DE_DE', 'anatomy'), ('DE_DE', 'business_ethics'), ('DE_DE', 'clinical_knowledge'), ('DE_DE', 'college_medicine'), ('DE_DE', 'global_facts'), ('DE_DE', 'human_aging'), ('DE_DE', 'management'), ('DE_DE', 'marketing'), ('DE_DE', 'medical_genetics'), ('DE_DE', 'miscellaneous'), ('DE_DE', 'nutrition'), ('DE_DE', 'professional_accounting'), ('DE_DE', 'professional_medicine'), ('DE_DE', 'virology')]
LANGUAGE = <Language.DEU: 'German'>
````

- Module: `eval_framework.tasks.benchmarks.mmmlu`

- File: [src/eval_framework/tasks/benchmarks/mmmlu.py](../../src/eval_framework/tasks/benchmarks/mmmlu.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/mmmlu.py)

- Link to dataset: [https://huggingface.co/datasets/openai/MMMLU](https://huggingface.co/datasets/openai/MMMLU)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "MMMLU_German"`.
