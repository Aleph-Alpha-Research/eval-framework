# MMLU_EU20_FR

````
NAME = MMLU_EU20_FR
DATASET_PATH = openGPT-X/mmlux
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = dev
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['abstract_algebra_FR', 'anatomy_FR', 'astronomy_FR', 'business_ethics_FR', 'clinical_knowledge_FR', 'college_biology_FR', 'college_chemistry_FR', 'college_computer_science_FR', 'college_mathematics_FR', 'college_medicine_FR', 'college_physics_FR', 'computer_security_FR', 'conceptual_physics_FR', 'econometrics_FR', 'electrical_engineering_FR', 'elementary_mathematics_FR', 'formal_logic_FR', 'global_facts_FR', 'high_school_biology_FR', 'high_school_chemistry_FR', 'high_school_computer_science_FR', 'high_school_european_history_FR', 'high_school_geography_FR', 'high_school_government_and_politics_FR', 'high_school_macroeconomics_FR', 'high_school_mathematics_FR', 'high_school_microeconomics_FR', 'high_school_physics_FR', 'high_school_psychology_FR', 'high_school_statistics_FR', 'high_school_us_history_FR', 'high_school_world_history_FR', 'human_aging_FR', 'human_sexuality_FR', 'international_law_FR', 'jurisprudence_FR', 'logical_fallacies_FR', 'machine_learning_FR', 'management_FR', 'marketing_FR', 'medical_genetics_FR', 'miscellaneous_FR', 'moral_disputes_FR', 'moral_scenarios_FR', 'nutrition_FR', 'philosophy_FR', 'prehistory_FR', 'professional_accounting_FR', 'professional_law_FR', 'professional_medicine_FR', 'professional_psychology_FR', 'public_relations_FR', 'security_studies_FR', 'sociology_FR', 'us_foreign_policy_FR', 'virology_FR', 'world_religions_FR']
LANGUAGE = <Language.FRA: 'French'>
````

- Module: [eval_framework.tasks.benchmarks.opengptx_eu20](eval_framework.tasks.benchmarks.opengptx_eu20)

- File: [src/eval_framework/tasks/benchmarks/opengptx_eu20.py](../../src/eval_framework/tasks/benchmarks/opengptx_eu20.py)

- Link to dataset: [https://huggingface.co/datasets/openGPT-X/mmlux](https://huggingface.co/datasets/openGPT-X/mmlux)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "MMLU_EU20_FR"`.
