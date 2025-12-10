# MMLU_EU20_DE

````
NAME = MMLU_EU20_DE
DATASET_PATH = openGPT-X/mmlux
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = dev
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['abstract_algebra_DE', 'anatomy_DE', 'astronomy_DE', 'business_ethics_DE', 'clinical_knowledge_DE', 'college_biology_DE', 'college_chemistry_DE', 'college_computer_science_DE', 'college_mathematics_DE', 'college_medicine_DE', 'college_physics_DE', 'computer_security_DE', 'conceptual_physics_DE', 'econometrics_DE', 'electrical_engineering_DE', 'elementary_mathematics_DE', 'formal_logic_DE', 'global_facts_DE', 'high_school_biology_DE', 'high_school_chemistry_DE', 'high_school_computer_science_DE', 'high_school_european_history_DE', 'high_school_geography_DE', 'high_school_government_and_politics_DE', 'high_school_macroeconomics_DE', 'high_school_mathematics_DE', 'high_school_microeconomics_DE', 'high_school_physics_DE', 'high_school_psychology_DE', 'high_school_statistics_DE', 'high_school_us_history_DE', 'high_school_world_history_DE', 'human_aging_DE', 'human_sexuality_DE', 'international_law_DE', 'jurisprudence_DE', 'logical_fallacies_DE', 'machine_learning_DE', 'management_DE', 'marketing_DE', 'medical_genetics_DE', 'miscellaneous_DE', 'moral_disputes_DE', 'moral_scenarios_DE', 'nutrition_DE', 'philosophy_DE', 'prehistory_DE', 'professional_accounting_DE', 'professional_law_DE', 'professional_medicine_DE', 'professional_psychology_DE', 'public_relations_DE', 'security_studies_DE', 'sociology_DE', 'us_foreign_policy_DE', 'virology_DE', 'world_religions_DE']
LANGUAGE = <Language.DEU: 'German'>
````

- Module: [eval_framework.tasks.benchmarks.opengptx_eu20](eval_framework.tasks.benchmarks.opengptx_eu20)

- File: [src/eval_framework/tasks/benchmarks/opengptx_eu20.py](../../src/eval_framework/tasks/benchmarks/opengptx_eu20.py)

- Link to dataset: [https://huggingface.co/datasets/openGPT-X/mmlux](https://huggingface.co/datasets/openGPT-X/mmlux)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "MMLU_EU20_DE"`.
