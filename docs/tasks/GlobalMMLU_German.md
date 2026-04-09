# GlobalMMLU_German

````
NAME = GlobalMMLU_German
DATASET_PATH = CohereLabs/Global-MMLU
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = dev
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
SUBJECTS = [('de', 'abstract_algebra'), ('de', 'astronomy'), ('de', 'college_biology'), ('de', 'college_chemistry'), ('de', 'college_computer_science'), ('de', 'college_mathematics'), ('de', 'college_physics'), ('de', 'computer_security'), ('de', 'conceptual_physics'), ('de', 'electrical_engineering'), ('de', 'elementary_mathematics'), ('de', 'high_school_biology'), ('de', 'high_school_chemistry'), ('de', 'high_school_computer_science'), ('de', 'high_school_mathematics'), ('de', 'high_school_physics'), ('de', 'high_school_statistics'), ('de', 'machine_learning'), ('de', 'formal_logic'), ('de', 'high_school_european_history'), ('de', 'high_school_us_history'), ('de', 'high_school_world_history'), ('de', 'international_law'), ('de', 'jurisprudence'), ('de', 'logical_fallacies'), ('de', 'moral_disputes'), ('de', 'moral_scenarios'), ('de', 'philosophy'), ('de', 'prehistory'), ('de', 'professional_law'), ('de', 'world_religions'), ('de', 'econometrics'), ('de', 'high_school_geography'), ('de', 'high_school_government_and_politics'), ('de', 'high_school_macroeconomics'), ('de', 'high_school_microeconomics'), ('de', 'high_school_psychology'), ('de', 'human_sexuality'), ('de', 'professional_psychology'), ('de', 'public_relations'), ('de', 'security_studies'), ('de', 'sociology'), ('de', 'us_foreign_policy'), ('de', 'anatomy'), ('de', 'business_ethics'), ('de', 'clinical_knowledge'), ('de', 'college_medicine'), ('de', 'global_facts'), ('de', 'human_aging'), ('de', 'management'), ('de', 'marketing'), ('de', 'medical_genetics'), ('de', 'miscellaneous'), ('de', 'nutrition'), ('de', 'professional_accounting'), ('de', 'professional_medicine'), ('de', 'virology')]
LANGUAGE = <Language.DEU: 'German'>
````

- Module: `eval_framework.tasks.benchmarks.global_mmlu`

- File: [src/eval_framework/tasks/benchmarks/global_mmlu.py](../../src/eval_framework/tasks/benchmarks/global_mmlu.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/global_mmlu.py)

- Link to dataset: [https://huggingface.co/datasets/CohereLabs/Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "GlobalMMLU_German"`.
