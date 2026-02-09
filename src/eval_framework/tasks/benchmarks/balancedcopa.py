from eval_framework.tasks.benchmarks.copa import COPA


class BalancedCOPA(COPA[str]):
    """COPA dataset: https://huggingface.co/datasets/pkavumba/balanced-copa"""

    NAME = "BalancedCOPA"
    DATASET_PATH = "aps/balanced_copa"
    # The dataset combines train and validation splits, so we use test split for evaluation.
    SAMPLE_SPLIT = "test"  # 500 examples
    FEWSHOT_SPLIT = "test"  # 500 examples
