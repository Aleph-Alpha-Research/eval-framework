from eval_framework.suite import SuiteAggregate, TaskSuite

HELLASWAG_RC = TaskSuite(name="hellaswag_rc_xlarge", tasks="HELLASWAG", num_fewshot=5)
WINOGRANDE_RC = TaskSuite(name="winogrande_rc_xlarge", tasks="WINOGRANDE", num_fewshot=5)
DROP_GEN = TaskSuite(name="drop_xlarge", tasks="DropCompletion_OLMES", num_fewshot=5)
NATURALQS_GEN = TaskSuite(name="naturalqs_xlarge", tasks="NaturalQsOpen", num_fewshot=5)
SQUAD_GEN = TaskSuite(name="squad_xlarge", tasks="SQuAD", num_fewshot=5)

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
        # HELLASWAG + WINOGRANDE share "Average Accuracy Loglikelihood"; DROP/NQ/SQuAD are skipped.
        SuiteAggregate(name="macro_loglikelihood", metric="Average Accuracy Loglikelihood", method="mean"),
        # NaturalQsOpen + SQuAD share "Average F1"; others are skipped.
        SuiteAggregate(name="macro_f1", metric="Average F1", method="mean"),
        # DROP F1 and Exact Match are unique to drop_xlarge — surface directly via passthrough.
        SuiteAggregate(name="drop_xlarge", metric=["Average DROP F1", "Average Exact Match"], method="passthrough"),
        # Per-task scores.
        SuiteAggregate(name="hellaswag_rc_xlarge", metric="Average Accuracy Loglikelihood", method="passthrough"),
        SuiteAggregate(name="winogrande_rc_xlarge", metric="Average Accuracy Loglikelihood", method="passthrough"),
        SuiteAggregate(
            name="naturalqs_xlarge", metric=["Average F1", "Average Accuracy Completion"], method="passthrough"
        ),
        SuiteAggregate(name="squad_xlarge", metric=["Average F1", "Average Accuracy Completion"], method="passthrough"),
    ],
)
