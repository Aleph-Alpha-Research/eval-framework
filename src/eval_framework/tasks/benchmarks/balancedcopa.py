from datasets import Dataset

from eval_framework.tasks.base import SubjectType
from eval_framework.tasks.benchmarks.copa import COPA


def split_dataset_by_id_ranges(
    dataset: Dataset, id_column: str, ranges: list[tuple[int, int]]
) -> tuple[Dataset, Dataset]:
    """Split a dataset into two based on whether the id column falls within given ranges.

    Args:
        dataset: The dataset to split.
        id_column: The name of the column containing the id values.
        ranges: A list of (low, high) tuples defining inclusive ranges.
            Rows whose id is within any of these ranges go into the first split.

    Returns:

    """

    def in_any_range(id_value: int) -> bool:
        return any(low <= id_value <= high for low, high in ranges)

    in_indices = [i for i, id_val in enumerate(dataset[id_column]) if in_any_range(id_val)]
    not_in_indices = [i for i, id_val in enumerate(dataset[id_column]) if not in_any_range(id_val)]

    return dataset.select(in_indices), dataset.select(not_in_indices)


class BalancedCOPA(COPA):
    """Balanced-COPA dataset: https://huggingface.co/datasets/pkavumba/balanced-copa"""

    NAME = "BalancedCOPA"
    DATASET_PATH = "pkavumba/balanced-copa"
    HF_REVISION = "813bd03cd6e07d9bd8d7333896ad5d40abb95ea9"
    SUBJECTS = ["no_subject"]

    def _resplit_dataset_into_train_and_val(self) -> None:
        # We split the train data into train and validation splits so that
        # the validation split matches the validation split of the original COPA dataset.
        self.dataset["train"], self.dataset["validation"] = split_dataset_by_id_ranges(
            self.dataset["train"], "id", [(401, 500), (1401, 1500)]
        )

    def _load_dataset(self, subject: SubjectType) -> None:
        super()._load_dataset(subject)
        self._resplit_dataset_into_train_and_val()
