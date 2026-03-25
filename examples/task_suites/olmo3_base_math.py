from eval_framework.suite import SuiteAggregate, TaskSuite

_GSM8K_OLMO3_N8_V2 = TaskSuite(
    name="gsm8k_olmo3_n8_v2",
    tasks="GSM8K_OLMES",
    repeats=8,
    temperature=0.6,
    top_p=0.6,
    max_tokens=512,
)

MINERVA_MATH = TaskSuite(
    name="minerva_math_olmes_n4_v2",
    tasks="MATHMinerva_OLMES",
    repeats=4,
    temperature=0.6,
    top_p=0.6,
    max_tokens=1024,
    num_fewshot=4,
)

suite = TaskSuite(
    name="olmo3_base_math",
    tasks=[
        _GSM8K_OLMO3_N8_V2,
        MINERVA_MATH,
    ],
    aggregates=[
        SuiteAggregate(
            name="Math Average Accuracy",
            metric=["Average Accuracy Completion", "Average Exact Match (Flex)"],
            method="mean",
        ),
        SuiteAggregate(
            name="gsm8k_olmo3_n8_v2",
            metric="Average Accuracy Completion",
            method="passthrough",
        ),
        SuiteAggregate(
            name="minerva_math_olmes_n4_v2",
            metric="Average Exact Match (Flex)",
            method="passthrough",
        ),
    ],
)
