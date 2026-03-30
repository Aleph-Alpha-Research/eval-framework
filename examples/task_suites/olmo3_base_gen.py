from eval_framework.suite import SuiteAggregate, TaskSuite

HELLASWAG_RC = TaskSuite(name="hellaswag_rc_xlarge", tasks="HELLASWAG_OLMES", num_fewshot=5)
WINOGRANDE_RC = TaskSuite(name="winogrande_rc_xlarge", tasks="WINOGRANDECloze", num_fewshot=5)
DROP_GEN = TaskSuite(name="drop_xlarge", tasks="DropCompletion_OLMES", num_fewshot=5)
NATURALQS_GEN = TaskSuite(name="naturalqs_xlarge", tasks="NaturalQsOpen", num_fewshot=5)
SQUAD_GEN = TaskSuite(name="squad_xlarge", tasks="SQuAD_OLMES", num_fewshot=5)

suite = TaskSuite(
    name="olmo3_base_gen",
    tasks=[
        HELLASWAG_RC,
        WINOGRANDE_RC,
        DROP_GEN,
        NATURALQS_GEN,
        SQUAD_GEN,
    ],
    aggregates=[
        # Joint average
        SuiteAggregate(
            name="macro",
            metric=[
                "Average Accuracy Normalized Loglikelihood",
                "Average Partial Evaluation Accuracy",
                "Average DROP F1",
                "Average F1 SQuAD Normalized",
            ],
            method="mean",
        ),
        # Per-task scores.
        SuiteAggregate(
            name="hellaswag_rc_xlarge", metric="Average Accuracy Normalized Loglikelihood", method="passthrough"
        ),
        SuiteAggregate(name="winogrande_rc_xlarge", metric="Average Partial Evaluation Accuracy", method="passthrough"),
        SuiteAggregate(name="naturalqs_xlarge", metric="Average DROP F1", method="passthrough"),
        SuiteAggregate(name="drop_xlarge", metric="Average DROP F1", method="passthrough"),
        SuiteAggregate(name="squad_xlarge", metric="Average F1 SQuAD Normalized", method="passthrough"),
    ],
)
