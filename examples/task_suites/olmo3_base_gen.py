from eval_framework.suite import MetricSource, SuiteAggregate, TaskSuite

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
        # Joint average — each task uses a different metric name.
        SuiteAggregate(
            name="macro",
            method="mean",
            sources=[
                MetricSource(child="hellaswag_rc_xlarge", metric="Average Accuracy Normalized Loglikelihood"),
                MetricSource(child="winogrande_rc_xlarge", metric="Average Partial Evaluation Accuracy"),
                MetricSource(child="drop_xlarge", metric="Average DROP F1"),
                MetricSource(child="naturalqs_xlarge", metric="Average DROP F1"),
                MetricSource(child="squad_xlarge", metric="Average F1 SQuAD Normalized"),
            ],
        ),
        # Per-task scores.
        SuiteAggregate(
            name="hellaswag_rc_xlarge",
            sources=[MetricSource(child="hellaswag_rc_xlarge", metric="Average Accuracy Normalized Loglikelihood")],
        ),
        SuiteAggregate(
            name="winogrande_rc_xlarge",
            sources=[MetricSource(child="winogrande_rc_xlarge", metric="Average Partial Evaluation Accuracy")],
        ),
        SuiteAggregate(
            name="naturalqs_xlarge",
            sources=[MetricSource(child="naturalqs_xlarge", metric="Average DROP F1")],
        ),
        SuiteAggregate(
            name="drop_xlarge",
            sources=[MetricSource(child="drop_xlarge", metric="Average DROP F1")],
        ),
        SuiteAggregate(
            name="squad_xlarge",
            sources=[MetricSource(child="squad_xlarge", metric="Average F1 SQuAD Normalized")],
        ),
    ],
)
