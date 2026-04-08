from eval_framework.suite import MetricSource, SuiteAggregate, TaskSuite

MMLU_HUMANITIES_SUBJECTS: list[str] = [
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions",
]

MMLU_SOCIAL_SCIENCES_SUBJECTS: list[str] = [
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
]

MMLU_OTHER_SUBJECTS: list[str] = [
    "business_ethics",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "human_aging",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "professional_accounting",
    "professional_medicine",
    "virology",
]


_MMLU_HUM = TaskSuite(
    name="mmlu_humanities_mc_olmes",
    tasks="MMLU_OLMES",
    num_fewshot=5,
    task_subjects=MMLU_HUMANITIES_SUBJECTS,
)
_MMLU_SOC = TaskSuite(
    name="mmlu_social_sciences_mc_olmes",
    tasks="MMLU_OLMES",
    num_fewshot=5,
    task_subjects=MMLU_SOCIAL_SCIENCES_SUBJECTS,
)
_MMLU_OTHER = TaskSuite(
    name="mmlu_other_mc_olmes",
    tasks="MMLU_OLMES",
    num_fewshot=5,
    task_subjects=MMLU_OTHER_SUBJECTS,
)

_CSQA = TaskSuite(name="csqa_mc_xlarge", tasks="CommonsenseQAMC_OLMES", num_fewshot=5)
_PIQA = TaskSuite(name="piqa_mc_xlarge", tasks="PIQA_OLMES", num_fewshot=5)
_SOCIALIQA = TaskSuite(name="socialiqa_mc_xlarge", tasks="SocialIQAMC_OLMES", num_fewshot=5)
_DROP_MC = TaskSuite(name="drop_mc_gen2mc_xlarge", tasks="DropMC_OLMES", num_fewshot=5)
_NQ_MC = TaskSuite(name="naturalqs_mc_gen2mc_xlarge", tasks="NaturalQsOpenMC_OLMES", num_fewshot=5)

_METRIC = "Average Accuracy Loglikelihood"
_ALL_TASKS = [_MMLU_HUM, _MMLU_SOC, _MMLU_OTHER, _CSQA, _PIQA, _SOCIALIQA, _DROP_MC, _NQ_MC]

suite = TaskSuite(
    name="olmo3_base_nonstem_qa_mc",
    tasks=_ALL_TASKS,
    aggregates=[
        SuiteAggregate(
            name="macro",
            method="mean",
            sources=[MetricSource(child=task.name, metric=_METRIC) for task in _ALL_TASKS],
        ),
    ]
    + [
        SuiteAggregate(
            name=task.name,
            sources=[MetricSource(child=task.name, metric=_METRIC)],
        )
        for task in _ALL_TASKS
    ],
)
