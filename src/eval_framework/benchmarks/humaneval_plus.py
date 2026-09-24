"""HumanEvalPlus: HumanEval on the EvalPlus dataset (https://huggingface.co/datasets/evalplus/humanevalplus),
which adds far more test cases per problem. The prompt shape is identical to HumanEval's ``_V2`` variants, so
these reuse the builders exposed by the ``humaneval`` module; only the dataset (and the execution metric,
which needs numpy in the sandbox) differ.
"""

from eval_framework.benchmarks.humaneval import v2_bpb, v2_execution
from eval_framework.contract import Benchmark
from eval_framework.metrics.completion.code_assertion import HumanEvalPlusCodeCompletionAssertion
from eval_framework.tasks.dataset_loading import DatasetPolicy

HUMANEVAL_PLUS_DATASET_PATH = "evalplus/humanevalplus"


def humaneval_plus(dataset: DatasetPolicy | None = None) -> Benchmark:
    return v2_execution(
        "HumanEvalPlus",
        dataset_path=HUMANEVAL_PLUS_DATASET_PATH,
        metrics=[HumanEvalPlusCodeCompletionAssertion],  # numpy in the sandbox for some EvalPlus tests
        dataset=dataset,
    )


def humaneval_plus_bpb(dataset: DatasetPolicy | None = None) -> Benchmark:
    return v2_bpb("HumanEvalPlusBPB", dataset_path=HUMANEVAL_PLUS_DATASET_PATH, dataset=dataset)


HUMANEVAL_PLUS_BENCHMARKS: list[Benchmark] = [humaneval_plus(), humaneval_plus_bpb()]
