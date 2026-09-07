from eval_framework.metrics.completion.code_assertion import HumanEvalPlusCodeCompletionAssertion
from eval_framework.tasks.benchmarks.humaneval import HumanEval_OLMES_V2, HumanEvalBPB_V2
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE


class HumanEvalPlus(HumanEval_OLMES_V2):
    """HumanEvalPlus - code generation format (pass@1 via test execution).

    HumanEvalPlus extends HumanEval with far more test cases per problem, so only the dataset
    path is overridden.
    """

    NAME = "HumanEvalPlus"
    DATASET_PATH = "evalplus/humanevalplus"
    SAMPLE_SPLIT = "test"
    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    METRICS = [HumanEvalPlusCodeCompletionAssertion]


class HumanEvalPlusBPB(HumanEvalBPB_V2):
    """HumanEvalPlus variant that scores loglikelihood of the gold canonical solution."""

    NAME = "HumanEvalPlusBPB"
    DATASET_PATH = "evalplus/humanevalplus"
    SAMPLE_SPLIT = "test"
    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE
